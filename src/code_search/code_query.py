#!/usr/bin/env python
# ############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-18 18:23:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-20 09:00:49
# ############################################
"""
code_query.py  --  Natural-language query interface for the semantic code index.

Embeds natural-language questions with ``nomic-embed-text`` via Ollama,
retrieves the most relevant code chunks from the file-based Qdrant collection
built by ``code_indexer.py``, and asks a local LLM (via Ollama) to synthesise
an answer.

By default, runs as an interactive REPL.  Type your question at the prompt;
the LLM sees the full conversation history so follow-up questions work
naturally.  Special commands:

    new  /new       Clear conversation history and start a fresh topic.
    quit exit /q    Exit the program.

Usage
-----
    python code_query.py                         # interactive (default)
    python code_query.py --no-interactive "..."  # single-shot from CLI/script
    python code_query.py --code-snippets         # show source snippets too

Options
-------
    --no-interactive     Single-shot mode: answer one question and exit.
    --code-snippets      Show source code snippets in results (default: off).
    --index-dir   DIR    Same directory passed to code_indexer.py.
                         Default: ~/.code_index
    --collection  NAME   Qdrant collection name.  Default: code_index
    --top-k       N      Number of chunks to retrieve per question.  Default: 5
    --model       NAME   Ollama model tag.  Default: llama3:8b
    --ollama-url  URL    Ollama base URL.  Default: http://localhost:11434
    --no-llm             Skip LLM step; print retrieved chunk locations only.

Dependencies
------------
    pip install qdrant-client requests
    ollama pull nomic-embed-text
    ollama pull llama3:8b
"""

import argparse
import readline  # noqa: F401  — imported for side-effect: enables Emacs keys in input()
import sys
import textwrap
from pathlib import Path

import requests
from qdrant_client import QdrantClient


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
SNIPPET_LINES       = 40      # max lines shown when --code-snippets is set

# Visual formatting
RULE      = "─" * 72
THIN_RULE = "╌" * 72

# Interactive-mode special commands
CMD_NEW  = {"new", "/new"}
CMD_QUIT = {"quit", "exit", "/q"}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CodeRetriever:
    """Embed a query and retrieve the top-k matching chunks from Qdrant.

    :param index_dir:  Directory containing the Qdrant file-based storage.
    :param collection: Qdrant collection name.
    :param top_k:      Number of results to return.
    :param ollama_url: Base URL for the Ollama REST API.
    """

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

    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        """Embed *question* via Ollama and return the top-k payload dicts.

        Uses the ``search_query:`` task prefix required by nomic-embed-text
        for asymmetric retrieval (counterpart to ``search_document:`` used
        at index time).

        :param question: Natural-language question from the user.
        :return: List of payload dicts ordered by descending similarity.
                 Each dict contains ``filepath``, ``start_line``,
                 ``end_line``, ``kind``, ``name``, ``text``, and ``score``.
        """
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

        result = self._qdrant.query_points(
            collection_name=self._collection,
            query=embeddings[0],
            limit=self._top_k,
            with_payload=True,
        )

        hits = []
        for hit in result.points:
            payload = dict(hit.payload)
            payload["score"] = round(hit.score, 4)
            hits.append(payload)
        return hits


# ---------------------------------------------------------------------------
# LLM synthesiser
# ---------------------------------------------------------------------------

class LLMSynthesiser:
    """Ask a local Ollama model to explain retrieved code chunks.

    Maintains a conversation history so follow-up questions within the same
    session have context from prior turns.

    :param model:      Ollama model tag, e.g. ``llama3:8b``.
    :param ollama_url: Base URL for the Ollama REST API.
    """

    _SYSTEM = (
        "You are a concise assistant that answers questions about a personal "
        "Python and Bash codebase.  Answer in as few words as possible.  "
        "If the answer is a command or invocation, show it directly.  "
        "Always cite the file path and line numbers when referring to specific "
        "code.  If the provided excerpts do not answer the question, say so "
        "in one sentence."
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._model    = model
        self._chat_url = f"{ollama_url.rstrip('/')}/api/chat"
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear conversation history to start a new topic."""
        self._history = []

    # ------------------------------------------------------------------
    def explain(self, question: str, chunks: list[dict]) -> str:
        """Generate a synthesised answer given *question* and *chunks*.

        Appends the exchange to the internal history so follow-up questions
        have context from earlier turns.

        :param question: The user's natural-language question.
        :param chunks:   Retrieved code chunks (payload dicts from Qdrant).
        :return: LLM response text, or an error message string.
        """
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
                f"[LLM unavailable]  Could not connect to Ollama at "
                f"{self._chat_url}.  Is 'ollama serve' running?"
            )
        except requests.exceptions.Timeout:
            answer = "[LLM unavailable]  Ollama request timed out after 120s."
        except requests.exceptions.RequestException as exc:
            answer = f"[LLM error]  {exc}"

        self._history.append({"role": "assistant", "content": answer})
        return answer


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

class ResultPrinter:
    """Format and print query results to stdout.

    :param show_snippets: When True, print source code blocks under each hit.
    """

    def __init__(self, show_snippets: bool = False) -> None:
        self._show_snippets = show_snippets

    # ------------------------------------------------------------------
    def print_results(
        self,
        question: str,
        chunks: list[dict],
        explanation: str | None,
    ) -> None:
        """Print the full result block for one query.

        :param question:    The user's question.
        :param chunks:      Retrieved code chunks.
        :param explanation: LLM explanation string, or None if skipped.
        """
        print()
        print(RULE)
        print(f"  QUERY:  {question}")
        print(RULE)

        if not chunks:
            print("\n  No matching chunks found.  Try re-indexing or "
                  "rephrasing the question.\n")
            return

        print(f"\n  {len(chunks)} chunk(s) retrieved:\n")
        for i, chunk in enumerate(chunks, 1):
            self._print_chunk(i, chunk)

        if explanation is not None:
            print(RULE)
            print("  LLM EXPLANATION")
            print(RULE)
            print()
            for line in explanation.splitlines():
                if len(line) > 78:
                    print(textwrap.fill(line, width=78,
                                        subsequent_indent="  "))
                else:
                    print(line)
            print()

        print(RULE)
        print()

    # ------------------------------------------------------------------
    def _print_chunk(self, index: int, chunk: dict) -> None:
        """Print one chunk: always the location header, optionally the code.

        :param index: 1-based display index.
        :param chunk: Payload dict from Qdrant.
        """
        filepath = chunk["filepath"]
        start    = chunk["start_line"]
        end      = chunk["end_line"]
        kind     = chunk["kind"]
        name     = chunk.get("name") or "(unnamed)"
        score    = chunk.get("score", 0.0)

        print(f"  [{index}]  {filepath}")
        print(f"        Lines {start}–{end}  │  "
              f"{kind}: {name}  │  score: {score:.4f}")

        if self._show_snippets:
            print()
            lines        = chunk["text"].splitlines()
            truncated    = len(lines) > SNIPPET_LINES
            for lineno, line in enumerate(lines[:SNIPPET_LINES], start=start):
                print(f"  {lineno:>6} │ {line}")
            if truncated:
                omitted = len(lines) - SNIPPET_LINES
                print(f"         │ … ({omitted} more lines"
                      " — open file to see all)")

        print()
        print(f"  {THIN_RULE}")
        print()


# ---------------------------------------------------------------------------
# Querier  (top-level orchestrator)
# ---------------------------------------------------------------------------

class CodeQuerier:
    """Orchestrate retrieval, LLM synthesis, and result printing.

    :param index_dir:     Directory containing Qdrant storage + manifest.
    :param collection:    Qdrant collection name.
    :param top_k:         Number of chunks to retrieve per question.
    :param model:         Ollama model tag.
    :param ollama_url:    Ollama base URL.
    :param use_llm:       When False, skip LLM and show chunk locations only.
    :param show_snippets: When True, print source code blocks in output.
    """

    def __init__(
        self,
        index_dir: Path,
        collection: str     = COLLECTION_NAME,
        top_k: int          = DEFAULT_TOP_K,
        model: str          = DEFAULT_MODEL,
        ollama_url: str     = DEFAULT_OLLAMA_URL,
        use_llm: bool       = True,
        show_snippets: bool = False,
    ) -> None:
        self._use_llm = use_llm

        print("Connecting to index ...")
        self._retriever   = CodeRetriever(
            index_dir=index_dir,
            collection=collection,
            top_k=top_k,
            ollama_url=ollama_url,
        )
        self._synthesiser = (
            LLMSynthesiser(model=model, ollama_url=ollama_url)
            if use_llm else None
        )
        self._printer = ResultPrinter(show_snippets=show_snippets)

    # ------------------------------------------------------------------
    def query(self, question: str) -> None:
        """Retrieve chunks and print results for *question*.

        :param question: Natural-language question from the user.
        """
        chunks = self._retriever.retrieve(question)

        explanation: str | None = None
        if self._use_llm and self._synthesiser is not None:
            print("Asking LLM ...")
            explanation = self._synthesiser.explain(question, chunks)

        self._printer.print_results(question, chunks, explanation)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear LLM conversation history for a new topic.

        Safe to call even when LLM is disabled.
        """
        if self._synthesiser is not None:
            self._synthesiser.reset()

    # ------------------------------------------------------------------
    def run_interactive(self, first_question: str = "") -> None:
        """Enter the interactive REPL loop.

        Reads questions from stdin, maintains conversation history across
        turns, and honours the ``new``/``quit`` special commands.

        :param first_question: Optional question to answer before prompting
                               the user, e.g. one supplied on the CLI.
        """
        print()
        print(RULE)
        print("  Code Search  —  interactive mode")
        print(f"  Model: {DEFAULT_MODEL}   Index: {DEFAULT_INDEX_DIR}")
        print("  Commands:  new / /new   → fresh topic")
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
                print()
                print("  [conversation history cleared — new topic]")
                print()
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
        help="Optional opening question. In interactive mode it is answered "
             "first before the prompt appears. In --no-interactive mode it is "
             "the only question answered.",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Single-shot mode: answer one question and exit.",
    )
    p.add_argument(
        "--code-snippets",
        action="store_true",
        help="Show source code snippets alongside chunk locations (default: off).",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        metavar="DIR",
        help=f"Index directory (same as code_indexer.py). "
             f"Default: {DEFAULT_INDEX_DIR}",
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
        help="Skip LLM step; show chunk locations only.",
    )
    return p


def main() -> None:
    """Entry point for the query CLI."""
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
            show_snippets = args.code_snippets,
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