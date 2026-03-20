#!/usr/bin/env python
# #############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-20 09:30:31
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-20 09:32:44
# #############################################

"""
code_search_server.py  --  HTTP API server for the semantic code search index.

Exposes a single ``POST /query`` endpoint.  The client sends a question and
the current conversation history; the server runs retrieval + LLM synthesis
and returns the answer plus the updated history.  All conversation state lives
on the client — the server is fully stateless.

Start with Gunicorn (recommended)::

    gunicorn -w 1 -b 0.0.0.0:58008 code_search_server:app

Or for quick testing::

    python code_search_server.py          # uses port 58008
    python code_search_server.py --port 12345

Request
-------
    POST /query
    Content-Type: application/json

    {
        "question": "where is the sunset calculation?",
        "history":  [                          # may be [] on first turn
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Response
--------
    200 OK
    {
        "answer":   "The sunset calculation is in …",
        "chunks":   [ { "filepath": "...", "start_line": 10,
                        "end_line": 40, "kind": "function",
                        "name": "calc_sunset", "score": 0.82 } ],
        "history":  [ ... ]          # updated history to send on next turn
    }

    4xx / 5xx
    { "error": "human-readable message" }

Dependencies
------------
    pip install flask gunicorn qdrant-client requests
"""

import argparse
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# Re-use the retriever and synthesiser from code_query.py which must be on
# the Python path (or in the same directory).
from code_query import (
    COLLECTION_NAME,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    DEFAULT_INDEX_DIR,
    OLLAMA_EMBED_MODEL,
    CodeRetriever,
)

import requests as _requests  # avoid shadowing Flask's `request`


# ---------------------------------------------------------------------------
# Configuration  (overridable via environment variables for Gunicorn setups)
# ---------------------------------------------------------------------------

import os

INDEX_DIR   = Path(os.environ.get("CODE_INDEX_DIR",   str(DEFAULT_INDEX_DIR))).expanduser()
COLLECTION  = os.environ.get("CODE_COLLECTION",       COLLECTION_NAME)
OLLAMA_URL  = os.environ.get("CODE_OLLAMA_URL",       DEFAULT_OLLAMA_URL)
LLM_MODEL   = os.environ.get("CODE_LLM_MODEL",        DEFAULT_MODEL)
TOP_K       = int(os.environ.get("CODE_TOP_K",        str(DEFAULT_TOP_K)))
DEFAULT_PORT = 58008

# System prompt — matches the one in LLMSynthesiser but lives here so the
# server controls it independently of the client.
_SYSTEM = (
    "You are a concise assistant that answers questions about a personal "
    "Python and Bash codebase.  Answer in as few words as possible.  "
    "If the answer is a command or invocation, show it directly.  "
    "Always cite the file path and line numbers when referring to specific "
    "code.  If the provided excerpts do not answer the question, say so "
    "in one sentence."
)

# ---------------------------------------------------------------------------
# Flask app + lazy-initialised retriever
# ---------------------------------------------------------------------------

app = Flask(__name__)
_retriever: CodeRetriever | None = None


def _get_retriever() -> CodeRetriever:
    """Return the shared CodeRetriever, initialising it on first call.

    :return: Initialised ``CodeRetriever`` instance.
    :raises RuntimeError: If the index directory does not exist.
    """
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
# Helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a text block for the LLM prompt.

    :param chunks: List of payload dicts from Qdrant.
    :return: Multi-line string with numbered excerpts.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] {chunk['filepath']}  "
            f"lines {chunk['start_line']}–{chunk['end_line']}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _call_llm(history: list[dict]) -> str:
    """Send *history* to Ollama and return the assistant reply.

    :param history: List of ``{role, content}`` dicts representing the full
                    conversation so far, including the new user turn.
    :return: Assistant response text, or an error string.
    """
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
    """Health-check endpoint.

    :return: JSON status object.
    """
    return jsonify({
        "status":     "ok",
        "index_dir":  str(INDEX_DIR),
        "llm_model":  LLM_MODEL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "top_k":      TOP_K,
    })


@app.post("/query")
def query():
    """Answer one question and return the updated conversation history.

    Expects JSON body with ``question`` (str) and ``history`` (list).
    Returns JSON with ``answer``, ``chunks``, and updated ``history``.

    :return: JSON response.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required and must be non-empty."}), 400

    history: list[dict] = body.get("history", [])
    if not isinstance(history, list):
        return jsonify({"error": "Field 'history' must be a list."}), 400

    # Retrieval
    try:
        retriever = _get_retriever()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    chunks = retriever.retrieve(question)

    # Build user turn and append to history
    context      = _build_context(chunks)
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"RELEVANT CODE EXCERPTS:\n{context}\n\n"
        "ANSWER (be brief):"
    )
    history = list(history)          # copy — don't mutate caller's list
    history.append({"role": "user", "content": user_content})

    # LLM synthesis
    answer = _call_llm(history)
    history.append({"role": "assistant", "content": answer})

    # Strip 'text' from chunks before sending — potentially large and the
    # client doesn't need it (it has file paths + line numbers to open).
    slim_chunks = [
        {k: v for k, v in c.items() if k != "text"}
        for c in chunks
    ]

    return jsonify({
        "answer":  answer,
        "chunks":  slim_chunks,
        "history": history,
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
    """Entry point for direct execution (dev/testing only)."""
    parser = _build_parser()
    args   = parser.parse_args()
    print(f"Starting dev server on {args.host}:{args.port} — "
          f"use Gunicorn for production.")
    print(f"  Index : {INDEX_DIR}")
    print(f"  Model : {LLM_MODEL}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()