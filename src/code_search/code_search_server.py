#!/usr/bin/env python
# #############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-20 09:30:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-23 20:00:40
# #############################################

"""
code_search_server.py  --  HTTP API server for the semantic code search index.

Exposes a single ``POST /query`` endpoint.  The client sends a question and
the current conversation history; the server runs retrieval + LLM synthesis
and returns the answer plus the updated history.  All conversation state lives
on the client — the server is fully stateless.

Start with Gunicorn (recommended)::

    gunicorn -w 1 -b 0.0.0.0:58008 code_search_server:app

Dependencies
------------
    pip install flask gunicorn qdrant-client requests

A worker thread monitors all directories listed in the CODE_WATCH_DIRS 
environment variable once a minute. Upon finding a change, it pauses
query service, re-indexes, and resumes query serice. Incoming queries 
are queued during the indexing. 

The environment variable should be modified in the the systemd service
file on the server machine: /etc/systemd/system/code-search.service.
Afterwards:
   sudo systemctl daemon-reload
   sudo systemctl restart code-search

Use 
   sudo journalctl -u code-search -f
to see when most recent indexing occurred.
"""

import argparse
import gc
import logging
import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
import requests as _requests  # avoid shadowing Flask's `request`

# Re-use the retriever from code_query.py
from code_search.code_query import (
    COLLECTION_NAME,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    DEFAULT_INDEX_DIR,
    OLLAMA_EMBED_MODEL,
    CodeRetriever,
)

# Import indexer components for the background watcher
from code_search.code_indexer import (
    CodeIndexer,
    MtimeManifest,
    MANIFEST_FILENAME,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (overridable via environment variables for Gunicorn setups)
# ---------------------------------------------------------------------------

INDEX_DIR   = Path(os.environ.get("CODE_INDEX_DIR",   str(DEFAULT_INDEX_DIR))).expanduser()
COLLECTION  = os.environ.get("CODE_COLLECTION",       COLLECTION_NAME)
OLLAMA_URL  = os.environ.get("CODE_OLLAMA_URL",       DEFAULT_OLLAMA_URL)
LLM_MODEL   = os.environ.get("CODE_LLM_MODEL",        DEFAULT_MODEL)
TOP_K       = int(os.environ.get("CODE_TOP_K",        str(DEFAULT_TOP_K)))
DEFAULT_PORT = 58008

# Parse the colon-separated paths from the environment for the watcher thread
WATCH_DIRS_STR = os.environ.get("CODE_WATCH_DIRS", "")
WATCH_DIRS = [Path(p).expanduser().resolve() for p in WATCH_DIRS_STR.split(":") if p.strip()]

# Prompts
_SYSTEM = (
    "You are a concise assistant that answers questions about a personal "
    "Python and Bash codebase.  Answer in as few words as possible.  "
    "If the answer is a command or invocation, show it directly.  "
    "Always cite the file path and line numbers when referring to specific "
    "code.  If the provided excerpts do not answer the question, say so "
    "in one sentence."
)

_EXPAND_SYSTEM = (
    "You are a search query expander for a codebase. "
    "Given a user's question, output a single string combining the original "
    "question with 3-4 highly relevant technical synonyms, related terms, "
    "and rephrasings to maximize vector retrieval overlap. "
    "Return ONLY the expanded query string, with no introductory text or formatting."
)

# ---------------------------------------------------------------------------
# Flask app + Concurrency Setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
_retriever: CodeRetriever | None = None
indexing_lock = threading.Lock()


def _get_retriever() -> CodeRetriever:
    """Return the shared CodeRetriever, initialising it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = CodeRetriever(
            index_dir=INDEX_DIR,
            collection=COLLECTION,
            top_k=TOP_K,
            ollama_url=OLLAMA_URL,
        )
    return _retriever


# ---------------------------------------------------------------------------
# Background Watcher
# ---------------------------------------------------------------------------

def _needs_reindexing() -> bool:
    if not WATCH_DIRS:
        return False
        
    manifest_path = INDEX_DIR / MANIFEST_FILENAME
    manifest = MtimeManifest(manifest_path)
    
    for root in WATCH_DIRS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
                and d not in {"__pycache__", "node_modules", ".git", "venv",
                               ".venv", "env", "dist", "build", ".tox"}
            ]
            for fname in filenames:
                p = Path(dirpath) / fname
                if p.suffix in {".py", ".sh"}:
                    if manifest.is_stale(p):
                        return True
    return False

def _background_watcher():
    while True:
        time.sleep(60)
        if _needs_reindexing():
            logger.info("Changes detected! Pausing queries to re-index...")
            
            with indexing_lock:
                global _retriever
                if _retriever is not None:
                    _retriever._qdrant.close()
                    _retriever = None
                    gc.collect() 
                
                try:
                    indexer = CodeIndexer(
                        index_dir=INDEX_DIR,
                        collection=COLLECTION,
                        ollama_url=OLLAMA_URL,
                        force=False,
                        verbose=False
                    )
                    indexer.index_roots(WATCH_DIRS)
                except Exception as e:
                    logger.error(f"Indexing failed: {e}")
                finally:
                    if 'indexer' in locals():
                        indexer._qdrant.close()
                        del indexer
                        gc.collect()
                    
                logger.info("Indexing complete. Resuming queries.")

if WATCH_DIRS:
    logger.info(f"Starting background watcher for: {[str(p) for p in WATCH_DIRS]}")
    watcher_thread = threading.Thread(target=_background_watcher, daemon=True)
    watcher_thread.start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['filepath']}  "
            f"lines {chunk['start_line']}–{chunk['end_line']}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _call_llm_expand(question: str) -> str:
    """Use LLM to generate search synonyms to overcome vector vocabulary gaps."""
    messages = [
        {"role": "system", "content": _EXPAND_SYSTEM},
        {"role": "user", "content": question}
    ]
    try:
        resp = _requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model":    LLM_MODEL,
                "messages": messages,
                "stream":   False,
                "options":  {"temperature": 0.2},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except _requests.exceptions.RequestException:
        return question

def _call_llm(history: list[dict]) -> str:
    messages = [{"role": "system", "content": _SYSTEM}] + history
    try:
        resp = _requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model":    LLM_MODEL,
                "messages": messages,
                "stream":   False,
                "options":  {"temperature": 0.1},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except _requests.exceptions.ConnectionError:
        return f"[LLM unavailable] Cannot connect to Ollama at {OLLAMA_URL}."
    except _requests.exceptions.Timeout:
        return "[LLM unavailable] Ollama request timed out after 120 s."
    except _requests.exceptions.RequestException as exc:
        return f"[LLM error] {exc}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return jsonify({
        "status":     "ok",
        "index_dir":  str(INDEX_DIR),
        "llm_model":  LLM_MODEL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "top_k":      TOP_K,
        "watch_dirs": [str(d) for d in WATCH_DIRS],
    })


@app.post("/query")
def query():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required and must be non-empty."}), 400

    history: list[dict] = body.get("history", [])
    if not isinstance(history, list):
        return jsonify({"error": "Field 'history' must be a list."}), 400

    # 1. Expand the query for better retrieval
    expanded_query = _call_llm_expand(question)

    # 2. Retrieval (Wrapped in the lock to wait if indexing is happening)
    with indexing_lock:
        try:
            retriever = _get_retriever()
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503

        chunks = retriever.retrieve(expanded_query)

    # 3. Build user turn using the ORIGINAL question (prevents confusing the LLM)
    context      = _build_context(chunks)
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"RELEVANT CODE EXCERPTS:\n{context}\n\n"
        "ANSWER (be brief):"
    )
    history = list(history)
    history.append({"role": "user", "content": user_content})

    # 4. LLM synthesis
    answer = _call_llm(history)
    history.append({"role": "assistant", "content": answer})

    return jsonify({
        "answer":  answer,
        "chunks":  chunks,
        "history": history,
        "expanded_query": expanded_query,  # Pass it back for debugging clients
    })


# ---------------------------------------------------------------------------
# CLI  (for quick testing without Gunicorn)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Code search HTTP API server (use Gunicorn in production).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on. Default: {DEFAULT_PORT}",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind. Default: 0.0.0.0 (all interfaces)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    logger.info(f"Starting dev server on {args.host}:{args.port} — use Gunicorn for production.")
    logger.info(f"  Index : {INDEX_DIR}")
    logger.info(f"  Model : {LLM_MODEL}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()