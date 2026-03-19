#!/usr/bin/env python
# ############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-18 18:23:19
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-18 18:24:15
# ############################################

"""
code_query.py  --  Natural-language query interface for the semantic code index.

Embeds a natural-language question with ``nomic-embed-code``, retrieves the
most relevant code chunks from the file-based Qdrant collection built by
``code_indexer.py``, and asks a local Llama model (via Ollama) to synthesise
an answer.  Results are printed to stdout in three sections per hit:

    1. File path + line range
    2. Code snippet
    3. LLM explanation

Usage
-----
    python code_query.py "where is the sunset/before-after calculation?"

Options
-------
    --index-dir   DIR    Same directory passed to code_indexer.py.
                         Default: ~/.code_index
    --collection  NAME   Qdrant collection name.  Default: code_index
    --top-k       N      Number of chunks to retrieve.  Default: 5
    --model       NAME   Ollama model tag.  Default: llama3.2:11b
    --ollama-url  URL    Ollama base URL.  Default: http://localhost:11434
    --no-llm             Skip LLM step; print only paths and snippets.
    --snippet-lines N    Max lines of each snippet to print.  Default: 40

Dependencies
------------
    pip install qdrant-client sentence-transformers requests
    ollama pull llama3.2:11b   # one-time model download
"""

import argparse
import sys
import textwrap
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "nomic-ai/nomic-embed-code"
EMBEDDING_DIM = 768
COLLECTION_NAME = "code_index"
DEFAULT_INDEX_DIR = Path.home() / ".code_index"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:11b"
DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_LINES = 40

# Visual formatting
RULE = "─" * 72
THIN_RULE = "╌" * 72


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CodeRetriever:
    """Embed a query and retrieve the top-k matching chunks from Qdrant.

    :param index_dir:  Directory containing the Qdrant file-based storage.
    :param collection: Qdrant collection name.
    :param top_k:      Number of results to return.
    """

    def __init__(
        self,
        index_dir: Path,
        collection: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._collection = collection
        self._top_k = top_k

        qdrant_path = index_dir / "qdrant_storage"
        if not qdrant_path.exists():
            raise FileNotFoundError(
                f"No Qdrant storage found at '{qdrant_path}'.\n"
                "Run code_indexer.py first to build the index."
            )

        self._qdrant = QdrantClient(path=str(qdrant_path))
        self._model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        """Embed *question* and return the top-k payload dicts.

        The ``nomic-embed-code`` model uses a ``query:`` prefix for
        asymmetric retrieval (counterpart to the ``passage:`` prefix used
        at index time).

        :param question: Natural-language question from the user.
        :return: List of payload dicts ordered by descending similarity,
                 each containing ``filepath``, ``start_line``, ``end_line``,
                 ``kind``, ``name``, ``text``, and ``score``.
        """
        vector = self._model.encode(
            f"query: {question}",
            normalize_embeddings=True,
        ).tolist()

        hits = self._qdrant.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=self._top_k,
            with_payload=True,
        )

        results = []
        for hit in hits:
            payload = dict(hit.payload)
            payload["score"] = round(hit.score, 4)
            results.append(payload)
        return results


# ---------------------------------------------------------------------------
# LLM synthesiser
# ---------------------------------------------------------------------------

class LLMSynthesiser:
    """Ask a local Ollama model to explain retrieved code chunks.

    :param model:       Ollama model tag, e.g. ``llama3.2:11b``.
    :param ollama_url:  Base URL for the Ollama REST API.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self._model = model
        self._generate_url = f"{ollama_url.rstrip('/')}/api/generate"

    # ------------------------------------------------------------------
    def explain(self, question: str, chunks: list[dict]) -> str:
        """Generate a synthesised answer given *question* and *chunks*.

        :param question: The original natural-language question.
        :param chunks:   Retrieved code chunks (payload dicts).
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

        prompt = (
            "You are a helpful assistant that answers questions about a "
            "personal Python and Bash codebase.\n\n"
            "The following code excerpts were retrieved as the most relevant "
            "matches for the user's question.  Use them to give a concise, "
            "accurate answer.  Always cite the file path and line numbers "
            "when you refer to specific code.  If none of the excerpts "
            "actually answer the question, say so plainly.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED EXCERPTS:\n{context}\n\n"
            "ANSWER:"
        )

        try:
            response = requests.post(
                self._generate_url,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return (
                f"[LLM unavailable]  Could not connect to Ollama at "
                f"{self._generate_url}.  Is 'ollama serve' running?"
            )
        except requests.exceptions.Timeout:
            return "[LLM unavailable]  Ollama request timed out after 120s."
        except requests.exceptions.RequestException as exc:
            return f"[LLM error]  {exc}"


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

class ResultPrinter:
    """Format and print query results to stdout.

    :param snippet_lines: Maximum number of source lines to show per chunk.
    """

    def __init__(self, snippet_lines: int = DEFAULT_SNIPPET_LINES) -> None:
        self._snippet_lines = snippet_lines

    # ------------------------------------------------------------------
    def print_results(
        self,
        question: str,
        chunks: list[dict],
        explanation: str | None,
    ) -> None:
        """Print the full query result block.

        :param question:    The original user question.
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

        # ── Retrieved chunks ────────────────────────────────────────────
        print(f"\n  {len(chunks)} chunk(s) retrieved:\n")
        for i, chunk in enumerate(chunks, 1):
            self._print_chunk(i, chunk)

        # ── LLM explanation ─────────────────────────────────────────────
        if explanation is not None:
            print(RULE)
            print("  LLM EXPLANATION")
            print(RULE)
            print()
            # Wrap long lines for readability
            for line in explanation.splitlines():
                if len(line) > 78:
                    wrapped = textwrap.fill(
                        line, width=78, subsequent_indent="  "
                    )
                    print(wrapped)
                else:
                    print(line)
            print()

        print(RULE)
        print()

    # ------------------------------------------------------------------
    def _print_chunk(self, index: int, chunk: dict) -> None:
        """Print a single chunk with path, snippet, and metadata.

        :param index: 1-based display index.
        :param chunk: Payload dict from Qdrant.
        """
        filepath = chunk["filepath"]
        start = chunk["start_line"]
        end = chunk["end_line"]
        kind = chunk["kind"]
        name = chunk.get("name") or "(unnamed)"
        score = chunk.get("score", 0.0)
        text: str = chunk["text"]

        # ── Location header ──────────────────────────────────────────────
        print(f"  [{index}]  {filepath}")
        print(f"        Lines {start}–{end}  │  {kind}: {name}  │  score: {score:.4f}")
        print()

        # ── Snippet ──────────────────────────────────────────────────────
        lines = text.splitlines()
        truncated = len(lines) > self._snippet_lines
        display_lines = lines[: self._snippet_lines]

        for lineno, line in enumerate(display_lines, start=start):
            print(f"  {lineno:>6} │ {line}")

        if truncated:
            omitted = len(lines) - self._snippet_lines
            print(f"         │ … ({omitted} more lines — open file to see all)")

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
    :param top_k:         Number of chunks to retrieve.
    :param model:         Ollama model tag.
    :param ollama_url:    Ollama base URL.
    :param use_llm:       When False, skip LLM and print chunks only.
    :param snippet_lines: Max source lines to display per chunk.
    """

    def __init__(
        self,
        index_dir: Path,
        collection: str = COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        use_llm: bool = True,
        snippet_lines: int = DEFAULT_SNIPPET_LINES,
    ) -> None:
        self._use_llm = use_llm

        print("Loading embedding model …")
        self._retriever = CodeRetriever(
            index_dir=index_dir,
            collection=collection,
            top_k=top_k,
        )
        self._synthesiser = (
            LLMSynthesiser(model=model, ollama_url=ollama_url)
            if use_llm
            else None
        )
        self._printer = ResultPrinter(snippet_lines=snippet_lines)

    # ------------------------------------------------------------------
    def query(self, question: str) -> None:
        """Run a full query cycle for *question*.

        :param question: Natural-language question from the user.
        """
        chunks = self._retriever.retrieve(question)

        explanation: str | None = None
        if self._use_llm and self._synthesiser is not None:
            print("Asking LLM …")
            explanation = self._synthesiser.explain(question, chunks)

        self._printer.print_results(question, chunks, explanation)


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
        nargs="+",
        help="Natural-language question (quote it or pass as multiple words).",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        metavar="DIR",
        help=f"Index directory (same as used by code_indexer.py). "
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
        help=f"Number of chunks to retrieve. Default: {DEFAULT_TOP_K}",
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
        help="Skip LLM step; print retrieved chunks only (faster).",
    )
    p.add_argument(
        "--snippet-lines",
        type=int,
        default=DEFAULT_SNIPPET_LINES,
        metavar="N",
        help=f"Max source lines to display per chunk. Default: {DEFAULT_SNIPPET_LINES}",
    )
    return p


def main() -> None:
    """Entry point for the query CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    question = " ".join(args.question)
    index_dir = args.index_dir.expanduser().resolve()

    try:
        querier = CodeQuerier(
            index_dir=index_dir,
            collection=args.collection,
            top_k=args.top_k,
            model=args.model,
            ollama_url=args.ollama_url,
            use_llm=not args.no_llm,
            snippet_lines=args.snippet_lines,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    querier.query(question)


if __name__ == "__main__":
    main()