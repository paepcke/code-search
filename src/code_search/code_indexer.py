#!/usr/bin/env python
# ##############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-18 18:10:41
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-24 09:35:43
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
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Generator

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import requests
from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython
import tree_sitter_bash as tsbash

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
DEFAULT_OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "code_index"
DEFAULT_INDEX_DIR = Path.home() / ".code_index"
MANIFEST_FILENAME = "mtime_manifest.json"

PY_CHUNK_TYPES = {"function_definition", "class_definition"}
SH_CHUNK_TYPES = {"function_definition"}
MIN_CHUNK_CHARS = 15

# Minimum lines a docstring must have to earn its own standalone chunk.
# A single-line ":param x: …" block is not worth indexing alone; a
# multi-sentence description is.
MIN_DOCSTRING_LINES = 3

# Patterns that flag a statement as configuration-related.
_CONFIG_PATTERNS = re.compile(
    r"os\.environ\.get\s*\("                            # env-var reads
    r"|\.add_argument\s*\("                             # argparse flags
    r"|^[A-Z][A-Z0-9_]{2,}\s*=\s*"                     # ALL_CAPS constants …
    r"(?:[\"'\/]|Path|True|False|[A-Z][A-Z0-9_]{2,})"  # … with config-like RHS
)

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class CodeChunker:
    """Extract semantic chunks from Python and Bash source files.

    Produces up to five distinct chunk kinds per file:

    ``module_doc``
        The module-level docstring (Python) or leading block comment (Bash).
        Stands alone so that operational/configuration prose is always
        findable regardless of how the rest of the file is chunked.

    ``docstring``
        The docstring of a function or class, emitted as a standalone chunk
        when it contains enough prose to be meaningful on its own
        (≥ ``MIN_DOCSTRING_LINES`` lines).  Tagged with the parent's name so
        the LLM can trace it back.

    ``config``
        A window of lines centred on a configuration-related statement —
        ``os.environ.get``, ``argparse.add_argument``, or a
        ``MODULE_LEVEL_CONSTANT = …`` assignment — together with any
        immediately preceding comment block.  Captures the operational
        vocabulary (env-var names, flag names, default values) that
        docstrings sometimes omit.

    ``function`` / ``class``
        The full source of each top-level function or class definition,
        unchanged from the original behaviour.

    ``file_skeleton``
        A condensed view of the whole file: function/class names plus
        inline comments.  Module docstrings are *excluded* here because
        they already appear in their own ``module_doc`` chunk.
    """

    def __init__(self) -> None:
        self._py_parser = Parser(Language(tspython.language()))
        self._sh_parser = Parser(Language(tsbash.language()))

    def chunks_from_file(self, path: Path) -> list[dict]:
        """Return all chunks extracted from *path*.

        :param path: Source file to chunk (.py or .sh).
        :return: List of chunk dicts ready for embedding and upsert.
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
    # Language-specific entry points
    # ------------------------------------------------------------------

    def _chunk_python(self, source: bytes, path: Path) -> list[dict]:
        tree  = self._py_parser.parse(source)
        lines = source.decode("utf-8", errors="replace").splitlines()
        chunks: list[dict] = []

        # 1. Module docstring — own chunk
        mod_doc = self._extract_module_doc_python(tree.root_node, lines)
        if mod_doc:
            chunks.append(mod_doc)

        # 2. Config chunks (env vars, argparse flags, module constants)
        for cfg in self._extract_config_chunks(lines):
            chunks.append(cfg)

        # 3. Function / class bodies
        for node in self._walk_top_level(tree.root_node, PY_CHUNK_TYPES):
            body_chunk = self._node_to_chunk(node, lines)
            if body_chunk:
                chunks.append(body_chunk)
            # 4. Standalone docstring for this function/class
            ds = self._extract_docstring_chunk(node, lines)
            if ds:
                chunks.append(ds)

        # 5. File skeleton (module docstring lines excluded)
        skeleton = self._semantic_skeleton(
            tree, lines, "python", skip_module_doc=mod_doc is not None
        )
        if skeleton:
            chunks.insert(0, skeleton)

        return chunks

    def _chunk_bash(self, source: bytes, path: Path) -> list[dict]:
        tree  = self._sh_parser.parse(source)
        lines = source.decode("utf-8", errors="replace").splitlines()
        chunks: list[dict] = []

        # 1. Leading block comment as module_doc equivalent
        mod_doc = self._extract_module_doc_bash(lines)
        if mod_doc:
            chunks.append(mod_doc)

        # 2. Config lines (env-var reads / variable assignments)
        for cfg in self._extract_config_chunks(lines):
            chunks.append(cfg)

        # 3. Function bodies
        for node in self._walk_top_level(tree.root_node, SH_CHUNK_TYPES):
            chunk = self._node_to_chunk(node, lines)
            if chunk:
                chunks.append(chunk)

        # 4. File skeleton (module doc excluded)
        skeleton = self._semantic_skeleton(
            tree, lines, "bash", skip_module_doc=mod_doc is not None
        )
        if skeleton:
            chunks.insert(0, skeleton)

        return chunks

    # ------------------------------------------------------------------
    # Module-doc extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_module_doc_python(root: Node, lines: list[str]) -> dict | None:
        """Return the module-level docstring as a ``module_doc`` chunk.

        :param root: Root node of the parsed tree.
        :param lines: Source lines.
        :return: Chunk dict or ``None`` if no module docstring is present.
        """
        for child in root.children:
            if child.type == "expression_statement":
                if (len(child.children) == 1
                        and child.children[0].type == "string"):
                    start = child.start_point[0]
                    end   = child.end_point[0]
                    text  = "\n".join(lines[start : end + 1]).strip()
                    if len(text) >= MIN_CHUNK_CHARS:
                        return {
                            "text":       text,
                            "start_line": start + 1,
                            "end_line":   end + 1,
                            "kind":       "module_doc",
                            "name":       "module_docstring",
                        }
            # Module docstring must be the first non-trivial statement
            if child.type not in {"comment", "expression_statement"}:
                break
        return None

    @staticmethod
    def _extract_module_doc_bash(lines: list[str]) -> dict | None:
        """Return the leading comment block as a ``module_doc`` chunk.

        :param lines: Source lines.
        :return: Chunk dict or ``None`` if no leading comment block.
        """
        comment_lines: list[int] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_lines.append(i)
            elif stripped == "":
                if comment_lines:
                    # Allow one blank line gap inside a block
                    continue
            else:
                break

        if not comment_lines:
            return None

        start = comment_lines[0]
        end   = comment_lines[-1]
        text  = "\n".join(lines[start : end + 1]).strip()
        if len(text) < MIN_CHUNK_CHARS:
            return None

        return {
            "text":       text,
            "start_line": start + 1,
            "end_line":   end + 1,
            "kind":       "module_doc",
            "name":       "module_comment",
        }

    # ------------------------------------------------------------------
    # Per-function/class docstring extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_docstring_chunk(node: Node, lines: list[str]) -> dict | None:
        """Emit a standalone ``docstring`` chunk for *node* if the docstring
        is prose-rich enough to justify its own embedding.

        :param node: A ``function_definition`` or ``class_definition`` node.
        :param lines: Source lines.
        :return: Chunk dict or ``None``.
        """
        # Find the body node
        body = None
        for child in node.children:
            if child.type in ("block", "suite"):
                body = child
                break
        if body is None:
            return None

        # First statement of the body — must be an expression_statement
        # containing a string literal
        for child in body.children:
            if child.type == "expression_statement":
                if (len(child.children) == 1
                        and child.children[0].type == "string"):
                    start = child.start_point[0]
                    end   = child.end_point[0]
                    text  = "\n".join(lines[start : end + 1]).strip()
                    num_lines = end - start + 1
                    if num_lines < MIN_DOCSTRING_LINES:
                        return None
                    # Derive parent name
                    parent_name = ""
                    for c in node.children:
                        if c.type == "identifier":
                            parent_name = c.text.decode("utf-8", errors="replace")
                            break
                    kind_label = (
                        "class" if node.type == "class_definition" else "function"
                    )
                    return {
                        "text":       text,
                        "start_line": start + 1,
                        "end_line":   end + 1,
                        "kind":       "docstring",
                        "name":       f"{kind_label}:{parent_name}",
                    }
            # Only the very first non-trivial child counts
            if child.type not in {"comment", "newline", "indent"}:
                break
        return None

    # ------------------------------------------------------------------
    # Config chunk extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_config_chunks(lines: list[str]) -> list[dict]:
        """Yield ``config`` chunks centred on configuration-related statements.

        Each chunk contains the triggering line plus any immediately
        preceding comment block (up to 8 lines) and one trailing line of
        context.  Overlapping windows are merged so a dense block of
        constants produces one chunk rather than N almost-identical ones.

        :param lines: Source lines.
        :return: List of config chunk dicts.
        """
        # Collect indices of lines that match a config pattern
        trigger_indices: list[int] = []
        for i, line in enumerate(lines):
            if _CONFIG_PATTERNS.search(line):
                trigger_indices.append(i)

        if not trigger_indices:
            return []

        MAX_COMMENT_LOOKBACK = 8
        TRAIL = 1  # lines of context after the trigger

        # Build windows [start, end] around each trigger
        windows: list[tuple[int, int]] = []
        for idx in trigger_indices:
            # Walk back to collect the preceding comment block
            comment_start = idx
            for back in range(1, MAX_COMMENT_LOOKBACK + 1):
                prev = idx - back
                if prev < 0:
                    break
                stripped = lines[prev].strip()
                if stripped.startswith("#") or stripped == "":
                    comment_start = prev
                else:
                    break
            end = min(idx + TRAIL, len(lines) - 1)
            windows.append((comment_start, end))

        # Merge overlapping / adjacent windows
        merged: list[tuple[int, int]] = [windows[0]]
        for start, end in windows[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        chunks: list[dict] = []
        for start, end in merged:
            text = "\n".join(lines[start : end + 1]).strip()
            if len(text) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text":       text,
                    "start_line": start + 1,
                    "end_line":   end + 1,
                    "kind":       "config",
                    "name":       "configuration",
                })
        return chunks

    # ------------------------------------------------------------------
    # Shared helpers (unchanged contract, minor extension)
    # ------------------------------------------------------------------

    @staticmethod
    def _walk_top_level(root: Node, target_types: set[str]) -> Generator[Node, None, None]:
        for child in root.children:
            if child.type in target_types:
                yield child
            else:
                for grandchild in child.children:
                    if grandchild.type in target_types:
                        yield grandchild

    @staticmethod
    def _node_to_chunk(node: Node, lines: list[str]) -> dict | None:
        start = node.start_point[0]
        end = node.end_point[0]
        text = "\n".join(lines[start : end + 1])
        if len(text) < MIN_CHUNK_CHARS:
            return None

        name = ""
        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode("utf-8", errors="replace")
                break

        kind = "class" if node.type == "class_definition" else "function"
        return {
            "text":       text,
            "start_line": start + 1,
            "end_line":   end + 1,
            "kind":       kind,
            "name":       name,
        }

    @staticmethod
    def _semantic_skeleton(
        tree: Node,
        lines: list[str],
        lang: str,
        skip_module_doc: bool = False,
    ) -> dict | None:
        """Build a condensed file skeleton: function/class names + inline comments.

        Module docstring lines are excluded when *skip_module_doc* is True
        because they are already emitted as a ``module_doc`` chunk.

        :param tree: Parsed tree for the file.
        :param lines: Source lines.
        :param lang: ``"python"`` or ``"bash"``.
        :param skip_module_doc: If True, omit the module-level docstring lines.
        :return: Skeleton chunk dict or ``None``.
        """
        lines_to_keep: set[int] = set()

        # Determine which lines belong to the module docstring so we can
        # optionally exclude them from the skeleton.
        module_doc_lines: set[int] = set()
        if skip_module_doc and lang == "python":
            for child in tree.root_node.children:
                if child.type == "expression_statement":
                    if (len(child.children) == 1
                            and child.children[0].type == "string"):
                        module_doc_lines.update(
                            range(child.start_point[0], child.end_point[0] + 1)
                        )
                if child.type not in {"comment", "expression_statement"}:
                    break

        def walk(node: Node) -> None:
            if node.type == "comment":
                lines_to_keep.update(
                    range(node.start_point[0], node.end_point[0] + 1)
                )
            elif lang == "python" and node.type == "expression_statement":
                if len(node.children) == 1 and node.children[0].type == "string":
                    line_range = range(
                        node.start_point[0], node.end_point[0] + 1
                    )
                    if not skip_module_doc or not module_doc_lines.intersection(
                        line_range
                    ):
                        lines_to_keep.update(line_range)
            elif node.type in {"function_definition", "class_definition"}:
                for child in node.children:
                    if child.type == "identifier":
                        lines_to_keep.add(child.start_point[0])
                        break
                else:
                    lines_to_keep.add(node.start_point[0])
            for child in node.children:
                walk(child)

        walk(tree.root_node)

        if not lines_to_keep:
            return None

        skeleton_lines = [
            lines[i] for i in sorted(lines_to_keep) if i < len(lines)
        ]
        text = "\n".join(skeleton_lines).strip()

        if len(text) < MIN_CHUNK_CHARS:
            return None

        return {
            "text":       text,
            "start_line": 1,
            "end_line":   len(lines),
            "kind":       "file_skeleton",
            "name":       "semantic_summary",
        }

# ---------------------------------------------------------------------------
# Manifest (mtime tracking)
# ---------------------------------------------------------------------------
class MtimeManifest:
    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._data: dict[str, float] = {}
        if manifest_path.exists():
            try:
                self._data = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def is_stale(self, path: Path) -> bool:
        key = str(path)
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return False
        return self._data.get(key, -1.0) != current_mtime

    def update(self, path: Path) -> None:
        try:
            self._data[str(path)] = path.stat().st_mtime
        except OSError:
            pass

    def remove_missing(self, known_paths: set[str]) -> list[str]:
        gone = [p for p in self._data if p not in known_paths]
        for p in gone:
            del self._data[p]
        return gone

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
class CodeIndexer:
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

    def _check_ollama(self) -> None:
        try:
            resp = requests.get(
                self._embed_url.replace("/api/embed", "/api/tags"),
                timeout=5,
            )
            resp.raise_for_status()
            models_list = [m["name"] for m in resp.json().get("models", [])]
            available = any(
                m == OLLAMA_EMBED_MODEL or m.startswith(OLLAMA_EMBED_MODEL + ":")
                for m in models_list
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

    def index_roots(self, roots: list[Path]) -> None:
        all_py_sh: set[str] = set()
        to_index: list[Path] = []

        for root in roots:
            for path in self._iter_sources(root):
                all_py_sh.add(str(path))
                if self._force or self._manifest.is_stale(path):
                    to_index.append(path)

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

        chunk_buffer: list[dict] = []
        t0 = time.time()
        files_done = 0

        for path in to_index:
            if self._verbose:
                print(f"  {path}")

            self._delete_file_points(str(path))

            chunks = self._chunker.chunks_from_file(path)
            for chunk in chunks:
                chunk["filepath"] = str(path)
                chunk_buffer.append(chunk)

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

        if chunk_buffer:
            self._embed_and_upsert(chunk_buffer)

        self._manifest.save()
        elapsed = time.time() - t0
        print(f"Done. Indexed {files_done} file(s) in {elapsed:.1f}s.")

    def _iter_sources(self, root: Path) -> Generator[Path, None, None]:
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
                    yield p

    def _delete_file_points(self, filepath: str) -> None:
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

    def _embed_and_upsert(self, chunks: list[dict]) -> None:
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