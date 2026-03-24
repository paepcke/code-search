#!/usr/bin/env python
# ############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-18 18:23:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-23 20:17:57
# ############################################

"""
code_query.py  --  Natural-language query interface for the semantic code index.

Embeds natural-language questions with ``nomic-embed-text`` via Ollama,
generates a sparse TF-IDF keyword vector, retrieves and mathematically merges 
the most relevant hybrid chunks from Qdrant, and asks a local LLM to synthesise
an answer.

Usage
-----
    python code_query.py                         # interactive (default)
    python code_query.py "..."                   # answer then prompt
    python code_query.py --no-interactive "..."  # single-shot from CLI/script

Options
-------
    --show-sources       Print the file paths and metadata of retrieved chunks.
    --show-context       Print the exact text of the chunks fed to the LLM.
    --no-interactive     Single-shot mode: answer one question and exit.
    --index-dir   DIR    Same directory passed to code_indexer.py.
    --collection  NAME   Qdrant collection name.  Default: code_index
    --top-k       N      Number of chunks to retrieve per question.
    --model       NAME   Ollama model tag.  Default: llama3:8b
    --ollama-url  URL    Ollama base URL.  Default: http://localhost:11434
    --no-llm             Skip LLM step; prints sources/context automatically.
"""

import argparse
import hashlib
import re
import readline  # noqa: F401
import sys
import textwrap
from collections import Counter
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_EMBED_MODEL  = "nomic-embed-text"
EMBEDDING_DIM       = 768
COLLECTION_NAME     = "code_index"
DEFAULT_INDEX_DIR   = Path.home() / ".code_index"
DEFAULT_OLLAMA_URL  = "http://localhost:11434"
DEFAULT_MODEL       = "llama3:8b"
DEFAULT_TOP_K       = 5

RULE      = "─" * 72
THIN_RULE = "╌" * 72
CMD_NEW  = {"new", "/new"}
CMD_QUIT = {"quit", "exit", "/q"}

# ---------------------------------------------------------------------------
# Tokenizer / Sparse Vector Generator
# ---------------------------------------------------------------------------

def _to_sparse_vector(text: str) -> models.SparseVector | None:
    """Must perfectly match the hashing strategy in code_indexer.py"""
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    if not tokens:
        return None
        
    counts = Counter(tokens)
    indices_values = []
    
    for token, freq in counts.items():
        idx = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:8], 16)
        indices_values.append((idx, float(freq)))
        
    indices_values.sort(key=lambda x: x[0])
    
    return models.SparseVector(
        indices=[x[0] for x in indices_values],
        values=[x[1] for x in indices_values]
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CodeRetriever:
    """Embed a query and perform Hybrid Search (Dense + Sparse) in Qdrant."""

    def __init__(
        self,
        index_dir: Path,
        collection: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._collection = collection
        self._top_k      = top_k
        self._embed_url  = f"{ollama_url.rstrip('/')}/api/embed"

        qdrant_path = index_dir / "qdrant_storage"
        if not qdrant_path.exists():
            raise FileNotFoundError(
                f"No Qdrant storage found at '{qdrant_path}'.\n"
                "Run code_indexer.py first to build the index."
            )
        self._qdrant = QdrantClient(path=str(qdrant_path))

    def retrieve(self, question: str) -> list[dict]:
        # 1. Get Dense Vector
        try:
            resp = requests.post(
                self._embed_url,
                json={"model": OLLAMA_EMBED_MODEL,
                      "input": f"search_query: {question}"},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: embedding query failed — {exc}", file=sys.stderr)
            return []

        embeddings = resp.json().get("embeddings", [])
        if not embeddings:
            print("ERROR: Ollama returned no embeddings.", file=sys.stderr)
            return []
            
        dense_emb = embeddings[0]

        # 2. Get Sparse Vector (TF mapped to Qdrant's IDF)
        sparse_emb = _to_sparse_vector(question)

        # 3. Build Prefetch rules for Hybrid fusion
        prefetch = [
            models.Prefetch(
                query=dense_emb,
                using="dense",
                limit=self._top_k * 2,
            )
        ]
        
        if sparse_emb is not None:
            prefetch.append(
                models.Prefetch(
                    query=sparse_emb,
                    using="sparse",
                    limit=self._top_k * 2,
                )
            )

        # 4. Execute Hybrid Search using Reciprocal Rank Fusion (RRF)
        result = self._qdrant.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self._top_k,
            with_payload=True,
        )

        hits = []
        for hit in result.points:
            payload = dict(hit.payload)
            payload["score"] = round(hit.score, 4) if hit.score else 0.0
            hits.append(payload)
            
        return hits


# ---------------------------------------------------------------------------
# LLM synthesiser
# ---------------------------------------------------------------------------

class LLMSynthesiser:
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

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._model    = model
        self._chat_url = f"{ollama_url.rstrip('/')}/api/chat"
        self._history: list[dict] = []

    def reset(self) -> None:
        self._history = []

    def expand(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self._EXPAND_SYSTEM},
            {"role": "user", "content": question}
        ]
        try:
            resp = requests.post(
                self._chat_url,
                json={
                    "model":    self._model,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"temperature": 0.2},
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException:
            return question

    def explain(self, question: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = (
                f"[{i}] {chunk['filepath']}  "
                f"lines {chunk['start_line']}–{chunk['end_line']}"
            )
            context_parts.append(f"{header}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        user_content = (
            f"QUESTION:\n{question}\n\n"
            f"RELEVANT CODE EXCERPTS:\n{context}\n\n"
            "ANSWER (be brief):"
        )

        self._history.append({"role": "user", "content": user_content})
        messages = [{"role": "system", "content": self._SYSTEM}] + self._history

        try:
            resp = requests.post(
                self._chat_url,
                json={
                    "model":    self._model,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"temperature": 0.1},
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.ConnectionError:
            answer = (
                f"[LLM unavailable] Could not connect to Ollama at {self._chat_url}."
            )
        except requests.exceptions.Timeout:
            answer = "[LLM unavailable] Ollama request timed out after 120s."
        except requests.exceptions.RequestException as exc:
            answer = f"[LLM error] {exc}"

        self._history.append({"role": "assistant", "content": answer})
        return answer


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

class ResultPrinter:
    def __init__(self, show_sources: bool = False, show_context: bool = False) -> None:
        self._show_sources = show_sources
        self._show_context = show_context

    def print_results(
        self,
        question: str,
        expanded_query: str,
        chunks: list[dict],
        explanation: str | None,
    ) -> None:
        print()
        print(RULE)
        print(f"  QUERY:  {question}")
        
        if (self._show_sources or self._show_context) and expanded_query and expanded_query != question:
            print(f"  EXPANDED: {expanded_query}")
            
        print(RULE)

        if not chunks:
            print("\n  No matching chunks found.\n")
            return

        if explanation is not None:
            print("\n  ANSWER:\n")
            for line in explanation.splitlines():
                if len(line) > 78:
                    print(textwrap.fill(line, width=78, subsequent_indent="  "))
                else:
                    print(f"  {line}")
            print()

        if self._show_sources or self._show_context or explanation is None:
            print(RULE)
            print(f"  RETRIEVED CONTEXT ({len(chunks)} chunks)")
            print(RULE)
            
            for i, chunk in enumerate(chunks, 1):
                self._print_chunk(i, chunk)

        print(RULE)
        print()

    def _print_chunk(self, index: int, chunk: dict) -> None:
        filepath = chunk.get("filepath", "?")
        start    = chunk.get("start_line", "?")
        end      = chunk.get("end_line", "?")
        kind     = chunk.get("kind", "?")
        name     = chunk.get("name") or "(unnamed)"
        score    = chunk.get("score", 0.0)

        print(f"\n  [{index}] {filepath}")
        print(f"        Lines {start}–{end} │ {kind}: {name} │ score: {score:.4f}")

        if self._show_context and "text" in chunk:
            print()
            lines = chunk["text"].splitlines()
            start_idx = start if isinstance(start, int) else 1
            for lineno, line in enumerate(lines, start=start_idx):
                print(f"  {lineno:>6} │ {line}")
            print(f"\n  {THIN_RULE}")


# ---------------------------------------------------------------------------
# Querier
# ---------------------------------------------------------------------------

class CodeQuerier:
    def __init__(
        self,
        index_dir: Path,
        collection: str     = COLLECTION_NAME,
        top_k: int          = DEFAULT_TOP_K,
        model: str          = DEFAULT_MODEL,
        ollama_url: str     = DEFAULT_OLLAMA_URL,
        use_llm: bool       = True,
        show_sources: bool  = False,
        show_context: bool  = False,
    ) -> None:
        self._use_llm = use_llm

        print("Connecting to index ...")
        self._retriever = CodeRetriever(
            index_dir=index_dir,
            collection=collection,
            top_k=top_k,
            ollama_url=ollama_url,
        )
        self._synthesiser = (
            LLMSynthesiser(model=model, ollama_url=ollama_url)
            if use_llm else None
        )
        self._printer = ResultPrinter(
            show_sources=show_sources,
            show_context=show_context
        )

    def query(self, question: str) -> None:
        expanded_query = question
        
        if self._use_llm and self._synthesiser is not None:
            print("Expanding query ...")
            expanded_query = self._synthesiser.expand(question)

        chunks = self._retriever.retrieve(expanded_query)

        explanation: str | None = None
        if self._use_llm and self._synthesiser is not None:
            print("Asking LLM ...")
            explanation = self._synthesiser.explain(question, chunks)

        self._printer.print_results(question, expanded_query, chunks, explanation)

    def reset(self) -> None:
        if self._synthesiser is not None:
            self._synthesiser.reset()

    def run_interactive(self, first_question: str = "") -> None:
        print()
        print(RULE)
        print("  Code Search  —  interactive mode")
        print(f"  Model: {DEFAULT_MODEL}   Index: {DEFAULT_INDEX_DIR}")
        print("  Commands:  new / /new        → fresh topic")
        print("             quit / exit / /q  → exit")
        print(RULE)
        print()

        if first_question:
            self.query(first_question)

        while True:
            try:
                raw = input("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue

            if raw.lower() in CMD_QUIT:
                break

            if raw.lower() in CMD_NEW:
                self.reset()
                print("\n  [conversation history cleared — new topic]\n")
                continue

            self.query(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Natural-language query tool for the semantic code index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "question",
        nargs="*",
        help="Optional opening question.",
    )
    p.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the file paths and metadata of retrieved chunks.",
    )
    p.add_argument(
        "--show-context",
        action="store_true",
        help="Print the exact text of the chunks fed to the LLM.",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Single-shot mode: answer one question and exit.",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        metavar="DIR",
        help=f"Index directory. Default: {DEFAULT_INDEX_DIR}",
    )
    p.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        metavar="NAME",
        help=f"Qdrant collection name. Default: {COLLECTION_NAME}",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"Chunks to retrieve per question. Default: {DEFAULT_TOP_K}",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"Ollama model tag. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM step; prints sources/context automatically.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    index_dir = args.index_dir.expanduser().resolve()

    try:
        querier = CodeQuerier(
            index_dir     = index_dir,
            collection    = args.collection,
            top_k         = args.top_k,
            model         = args.model,
            ollama_url    = args.ollama_url,
            use_llm       = not args.no_llm,
            show_sources  = args.show_sources,
            show_context  = args.show_context,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.no_interactive:
        if not args.question:
            parser.error("--no-interactive requires a question argument.")
        querier.query(" ".join(args.question))
    else:
        first = " ".join(args.question) if args.question else ""
        querier.run_interactive(first_question=first)


if __name__ == "__main__":
    main()