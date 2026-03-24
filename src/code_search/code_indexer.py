#!/usr/bin/env python
# ##############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-18 18:10:41
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-23 17:05:23
# ##############################################

"""
code_indexer.py  --  Semantic code indexer for personal Python/Bash repositories.

NOTE: used by code_search_server.py. Do not run directly.
      See head comment of code_search_server.py for adding
      new code directories.

      The server automatically monitors a set of code directories
      that are set in /etc/systemd/system/code-search.service. 
      Re-indexing upon code changes occurs automatically. 

      Use 
         sudo journalctl -u code-search -f
      
      to see when most recent indexing occurred.

Walks one or more root directories, extracts function- and class-level chunks
from ``*.py`` and ``*.sh`` files using tree-sitter, embeds each chunk with
``nomic-embed-text`` via Ollama, and upserts them into a file-based Qdrant
collection.  Re-runs are incremental: only files whose mtime has changed
since the last run are re-processed.

Usage
-----
    python code_indexer.py  /path/to/repos  [/another/path ...]  [options]

Options
-------
    --index-dir   DIR    Directory that holds the Qdrant storage and the
                         mtime manifest.  Default: ~/.code_index
    --collection  NAME   Qdrant collection name.  Default: code_index
    --batch-size  N      Embedding batch size.  Default: 32
    --force              Ignore the mtime manifest and re-index everything.
    --verbose            Print each file as it is processed.

Dependencies
------------
    pip install qdrant-client requests tree-sitter \
                tree-sitter-python tree-sitter-bash
    ollama pull nomic-embed-text   # one-time model download
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Generator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
import requests
from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython
import tree_sitter_bash as tsbash


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768          # nomic-embed-text output dimension
DEFAULT_OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "code_index"
DEFAULT_INDEX_DIR = Path.home() / ".code_index"
MANIFEST_FILENAME = "mtime_manifest.json"

# tree-sitter node types that define a "chunk" boundary
PY_CHUNK_TYPES = {"function_definition", "class_definition"}
SH_CHUNK_TYPES = {"function_definition"}

# Minimum characters for a chunk to be worth embedding
MIN_CHUNK_CHARS = 40


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class CodeChunker:
    """Parse source files with tree-sitter and yield logical chunks.

    Each chunk is a dict with keys:
        text       -- the raw source text of the chunk
        start_line -- 1-based first line
        end_line   -- 1-based last line
        kind       -- 'function' | 'class' | 'module_preamble'
        name       -- identifier name when available, else ''
    """

    def __init__(self) -> None:
        self._py_parser = Parser(Language(tspython.language()))
        self._sh_parser = Parser(Language(tsbash.language()))

    # ------------------------------------------------------------------
    def chunks_from_file(self, path: Path) -> list[dict]:
        """Return a list of chunk dicts for *path*.

        :param path: Absolute path to a ``.py`` or ``.sh`` file.
        :return: List of chunk dicts, possibly empty.
        """
        try:
            source = path.read_bytes()
        except OSError:
            return []

        suffix = path.suffix.lower()
        if suffix == ".py":
            return self._chunk_python(source, path)
        if suffix == ".sh":
            return self._chunk_bash(source, path)
        return []

    # ------------------------------------------------------------------
    def _chunk_python(self, source: bytes, path: Path) -> list[dict]:
        tree = self._py_parser.parse(source)
        lines = source.decode("utf-8", errors="replace").splitlines()
        chunks: list[dict] = []
        top_level_covered: list[tuple[int, int]] = []

        for node in self._walk_top_level(tree.root_node, PY_CHUNK_TYPES):
            chunk = self._node_to_chunk(node, lines)
            if chunk:
                chunks.append(chunk)
                top_level_covered.append(
                    (node.start_point[0], node.end_point[0])
                )

        # Anything not inside a function/class becomes a module preamble chunk
        preamble = self._preamble_chunk(lines, top_level_covered)
        if preamble:
            chunks.insert(0, preamble)

        return chunks

    # ------------------------------------------------------------------
    def _chunk_bash(self, source: bytes, path: Path) -> list[dict]:
        tree = self._sh_parser.parse(source)
        lines = source.decode("utf-8", errors="replace").splitlines()
        chunks: list[dict] = []
        covered: list[tuple[int, int]] = []

        for node in self._walk_top_level(tree.root_node, SH_CHUNK_TYPES):
            chunk = self._node_to_chunk(node, lines)
            if chunk:
                chunks.append(chunk)
                covered.append((node.start_point[0], node.end_point[0]))

        preamble = self._preamble_chunk(lines, covered)
        if preamble:
            chunks.insert(0, preamble)

        return chunks

    # ------------------------------------------------------------------
    @staticmethod
    def _walk_top_level(
        root: Node, target_types: set[str]
    ) -> Generator[Node, None, None]:
        """Yield direct children (and their children one level deep) that
        match *target_types*.  We avoid deeply nested recursion so that
        inner functions stay attached to their parent chunk.
        """
        for child in root.children:
            if child.type in target_types:
                yield child
            else:
                # One level deeper (e.g. decorated functions)
                for grandchild in child.children:
                    if grandchild.type in target_types:
                        yield grandchild

    # ------------------------------------------------------------------
    @staticmethod
    def _node_to_chunk(node: Node, lines: list[str]) -> dict | None:
        start = node.start_point[0]   # 0-based
        end = node.end_point[0]       # 0-based inclusive
        text = "\n".join(lines[start : end + 1])
        if len(text) < MIN_CHUNK_CHARS:
            return None

        # Try to extract the identifier name
        name = ""
        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8", errors="replace")
                break

        kind = "class" if node.type == "class_definition" else "function"
        return {
            "text": text,
            "start_line": start + 1,
            "end_line": end + 1,
            "kind": kind,
            "name": name,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _preamble_chunk(
        lines: list[str], covered: list[tuple[int, int]]
    ) -> dict | None:
        """Return a chunk for all lines NOT covered by any function/class."""
        covered_set: set[int] = set()
        for start, end in covered:
            covered_set.update(range(start, end + 1))

        preamble_lines = [
            line
            for i, line in enumerate(lines)
            if i not in covered_set
        ]
        text = "\n".join(preamble_lines).strip()
        if len(text) < MIN_CHUNK_CHARS:
            return None
        return {
            "text": text,
            "start_line": 1,
            "end_line": len(lines),
            "kind": "module_preamble",
            "name": "",
        }


# ---------------------------------------------------------------------------
# Manifest (mtime tracking)
# ---------------------------------------------------------------------------

class MtimeManifest:
    """Persist a ``{filepath: mtime}`` mapping to disk.

    :param manifest_path: Full path to the JSON manifest file.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._data: dict[str, float] = {}
        if manifest_path.exists():
            try:
                self._data = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def is_stale(self, path: Path) -> bool:
        """Return True if *path* is new or has been modified since last index.

        :param path: File to check.
        :return: True when re-indexing is needed.
        """
        key = str(path)
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return False
        return self._data.get(key, -1.0) != current_mtime

    def update(self, path: Path) -> None:
        """Record the current mtime of *path*.

        :param path: File that was just indexed.
        """
        try:
            self._data[str(path)] = path.stat().st_mtime
        except OSError:
            pass

    def remove_missing(self, known_paths: set[str]) -> list[str]:
        """Drop manifest entries for files that no longer exist.

        :param known_paths: Set of string paths currently on disk.
        :return: List of paths that were removed from the manifest.
        """
        gone = [p for p in self._data if p not in known_paths]
        for p in gone:
            del self._data[p]
        return gone

    def save(self) -> None:
        """Flush the manifest to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class CodeIndexer:
    """Orchestrate chunking, embedding, and Qdrant upsert.

    :param index_dir:    Directory for Qdrant storage and the mtime manifest.
    :param collection:   Qdrant collection name.
    :param batch_size:   Number of chunks to embed per Ollama request.
    :param ollama_url:   Base URL for the Ollama REST API.
    :param force:        When True, ignore mtime manifest and re-index all.
    :param verbose:      When True, print each file as it is processed.
    """

    def __init__(
        self,
        index_dir: Path,
        collection: str = COLLECTION_NAME,
        batch_size: int = 32,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        self._index_dir = index_dir
        self._collection = collection
        self._batch_size = batch_size
        self._embed_url = f"{ollama_url.rstrip('/')}/api/embed"
        self._force = force
        self._verbose = verbose

        index_dir.mkdir(parents=True, exist_ok=True)

        print(f"Embedding model : {OLLAMA_EMBED_MODEL} via Ollama")
        self._check_ollama()

        print("Opening Qdrant storage ...")
        self._qdrant = QdrantClient(path=str(index_dir / "qdrant_storage"))
        self._ensure_collection()

        self._chunker = CodeChunker()
        self._manifest = MtimeManifest(index_dir / MANIFEST_FILENAME)

    # ------------------------------------------------------------------
    def _check_ollama(self) -> None:
        """Verify Ollama is reachable and the embed model is available.

        :raises SystemExit: If Ollama cannot be reached or the model is absent.
        """
        try:
            resp = requests.get(
                self._embed_url.replace("/api/embed", "/api/tags"),
                timeout=5,
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama tags may include ":latest" suffix
            available = any(
                m == OLLAMA_EMBED_MODEL or m.startswith(OLLAMA_EMBED_MODEL + ":")
                for m in models
            )
            if not available:
                print(
                    f"ERROR: model '{OLLAMA_EMBED_MODEL}' not found in Ollama.\n"
                    f"Run:  ollama pull {OLLAMA_EMBED_MODEL}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"  Ollama reachable, model '{OLLAMA_EMBED_MODEL}' present.")
        except requests.exceptions.ConnectionError:
            print(
                f"ERROR: Cannot connect to Ollama at {self._embed_url}.\n"
                "Is 'ollama serve' running?",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self._collection not in existing:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            print(f"Created Qdrant collection '{self._collection}'.")
        else:
            print(f"Using existing Qdrant collection '{self._collection}'.")

    # ------------------------------------------------------------------
    def index_roots(self, roots: list[Path]) -> None:
        """Walk *roots*, index changed files, remove deleted files.

        :param roots: List of directory paths to walk recursively.
        """
        all_py_sh: set[str] = set()
        to_index: list[Path] = []

        for root in roots:
            for path in self._iter_sources(root):
                all_py_sh.add(str(path))
                if self._force or self._manifest.is_stale(path):
                    to_index.append(path)

        # Remove stale Qdrant points for deleted files
        gone = self._manifest.remove_missing(all_py_sh)
        if gone:
            print(f"Removing {len(gone)} deleted file(s) from index …")
            for filepath in gone:
                self._delete_file_points(filepath)

        if not to_index:
            print("Index is up to date. Nothing to do.")
            self._manifest.save()
            return

        print(
            f"Indexing {len(to_index)} file(s) "
            f"({len(all_py_sh) - len(to_index)} unchanged) …"
        )

        chunk_buffer: list[dict] = []   # accumulate across files
        t0 = time.time()
        files_done = 0

        for path in to_index:
            if self._verbose:
                print(f"  {path}")

            # Delete existing points for this file before re-indexing
            self._delete_file_points(str(path))

            chunks = self._chunker.chunks_from_file(path)
            for chunk in chunks:
                chunk["filepath"] = str(path)
                chunk_buffer.append(chunk)

            # Flush in batches to keep memory manageable
            while len(chunk_buffer) >= self._batch_size:
                self._embed_and_upsert(chunk_buffer[: self._batch_size])
                chunk_buffer = chunk_buffer[self._batch_size :]

            self._manifest.update(path)
            files_done += 1

            if files_done % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  … {files_done}/{len(to_index)} files  "
                    f"({elapsed:.0f}s elapsed)"
                )

        # Flush remainder
        if chunk_buffer:
            self._embed_and_upsert(chunk_buffer)

        self._manifest.save()
        elapsed = time.time() - t0
        print(f"Done. Indexed {files_done} file(s) in {elapsed:.1f}s.")

    # ------------------------------------------------------------------
    def _iter_sources(self, root: Path) -> Generator[Path, None, None]:
        """Yield all ``*.py`` and ``*.sh`` files under *root*.

        :param root: Directory to walk.
        """
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden dirs and common noise dirs
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
                and d not in {"__pycache__", "node_modules", ".git", "venv",
                               ".venv", "env", "dist", "build", ".tox"}
            ]
            for fname in filenames:
                p = Path(dirpath) / fname
                if p.suffix in {".py", ".sh"}:
                    yield p

    # ------------------------------------------------------------------
    def _delete_file_points(self, filepath: str) -> None:
        """Remove all Qdrant points whose payload filepath matches.

        :param filepath: String path used as the payload key.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self._qdrant.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="filepath",
                        match=MatchValue(value=filepath),
                    )
                ]
            ),
        )

    # ------------------------------------------------------------------
    def _embed_and_upsert(self, chunks: list[dict]) -> None:
        """Embed *chunks* via Ollama and upsert them into Qdrant.

        :param chunks: List of chunk dicts (must include 'filepath').
        """
        # nomic-embed-text uses task prefixes for asymmetric retrieval.
        # Code passages use the 'search_document' prefix at index time.
        texts = [f"search_document: {c['text']}" for c in chunks]

        try:
            response = requests.post(
                self._embed_url,
                json={"model": OLLAMA_EMBED_MODEL, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"WARNING: embedding request failed — {exc}. Skipping batch.")
            return

        vectors = response.json().get("embeddings", [])
        if len(vectors) != len(chunks):
            print(
                f"WARNING: expected {len(chunks)} embeddings, "
                f"got {len(vectors)}. Skipping batch."
            )
            return

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "filepath": chunk["filepath"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "kind": chunk["kind"],
                    "name": chunk["name"],
                    "text": chunk["text"],
                },
            )
            for vec, chunk in zip(vectors, chunks)
        ]
        self._qdrant.upsert(collection_name=self._collection, points=points)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Semantic code indexer for Python/Bash repositories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "roots",
        nargs="+",
        metavar="DIR",
        help="One or more root directories to index recursively.",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        metavar="DIR",
        help=f"Storage directory for Qdrant + manifest. Default: {DEFAULT_INDEX_DIR}",
    )
    p.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        metavar="NAME",
        help=f"Qdrant collection name. Default: {COLLECTION_NAME}",
    )
    p.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Embedding batch size. Default: 32",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore mtime manifest and re-index all files.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each file path as it is processed.",
    )
    return p


def main() -> None:
    """Entry point for the indexer CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    for root in roots:
        if not root.is_dir():
            print(f"ERROR: '{root}' is not a directory.", file=sys.stderr)
            sys.exit(1)

    indexer = CodeIndexer(
        index_dir=args.index_dir.expanduser().resolve(),
        collection=args.collection,
        batch_size=args.batch_size,
        ollama_url=args.ollama_url,
        force=args.force,
        verbose=args.verbose,
    )
    indexer.index_roots(roots)


if __name__ == "__main__":
    main()