#!/usr/bin/env python
# ############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-20 09:31:24
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-23 18:03:56
# #############################################

"""
code_search_client.py  --  Interactive client for the remote code search server.

Connects to a ``code_search_server.py`` instance running on a remote host
(e.g. sextus) and provides the same interactive REPL experience as running
``code_query.py`` locally.  Conversation history is maintained here on the
client and sent with every request.

By default, it outputs ONLY the LLM's answer.

Usage
-----
    python code_search_client.py                        # connect to sextus:58008
    python code_search_client.py --host 10.0.0.5        # explicit IP
    python code_search_client.py --host sextus.local    # mDNS name
    python code_search_client.py "opening question"     # answer then prompt
    python code_search_client.py --no-interactive "..." # single-shot

Options
-------
    --host   HOST     Hostname or IP of the server.  Default: sextus.local
    --port   N        Server port.                   Default: 58008
    --show-sources    Print the file paths and metadata of retrieved chunks.
    --show-context    Print the exact text of the chunks fed to the LLM.
    --no-interactive  Single-shot mode.

Special commands (interactive mode)
------------------------------------
    new  /new       Clear conversation history — start a fresh topic.
    quit exit /q    Exit.

Dependencies
------------
    pip install requests
"""

import argparse
import readline  # noqa: F401
import sys
import textwrap

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HOST = "sextus.local"
DEFAULT_PORT = 58008

RULE      = "─" * 72
THIN_RULE = "╌" * 72

CMD_NEW  = {"new", "/new"}
CMD_QUIT = {"quit", "exit", "/q"}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CodeSearchClient:
    """HTTP client for the code search server."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        show_sources: bool = False,
        show_context: bool = False,
    ) -> None:
        self._base_url     = f"http://{host}:{port}"
        self._show_sources = show_sources
        self._show_context = show_context
        self._history: list[dict] = []
        self._check_server()

    def _check_server(self) -> None:
        try:
            resp = requests.get(f"{self._base_url}/", timeout=5)
            resp.raise_for_status()
            info = resp.json()
            print(f"Connected to {self._base_url}")
            print(f"  Index : {info.get('index_dir', '?')}")
            print(f"  Model : {info.get('llm_model', '?')}")
        except requests.exceptions.ConnectionError:
            print(
                f"ERROR: Cannot connect to code search server at "
                f"{self._base_url}.\n"
                f"Is 'gunicorn … code_search_server:app' running on the server?",
                file=sys.stderr,
            )
            sys.exit(1)
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: Server health check failed — {exc}", file=sys.stderr)
            sys.exit(1)

    def reset(self) -> None:
        self._history = []

    def query(self, question: str) -> None:
        print("Asking server ...")
        try:
            resp = requests.post(
                f"{self._base_url}/query",
                json={"question": question, "history": self._history},
                timeout=180,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            print("ERROR: Server timed out after 180 s.", file=sys.stderr)
            return
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return

        data = resp.json()

        if "error" in data:
            print(f"Server error: {data['error']}", file=sys.stderr)
            return

        self._history = data.get("history", self._history)

        self._print_results(
            question=question,
            chunks=data.get("chunks", []),
            answer=data.get("answer", ""),
        )

    def _print_results(
        self,
        question: str,
        chunks: list[dict],
        answer: str,
    ) -> None:
        print()
        print(RULE)
        print(f"  QUERY:  {question}")
        print(RULE)

        if not chunks:
            print("\n  No matching chunks found.\n")
            return

        # Print LLM Answer First
        if answer:
            print("\n  ANSWER:\n")
            for line in answer.splitlines():
                if len(line) > 78:
                    print(textwrap.fill(line, width=78, subsequent_indent="  "))
                else:
                    print(f"  {line}")
            print()

        # Print Sources / Context if requested (or if no answer was generated)
        if self._show_sources or self._show_context or not answer:
            print(RULE)
            print(f"  RETRIEVED CONTEXT ({len(chunks)} chunks)")
            print(RULE)
            
            for i, chunk in enumerate(chunks, 1):
                filepath = chunk.get("filepath", "?")
                start    = chunk.get("start_line", "?")
                end      = chunk.get("end_line", "?")
                kind     = chunk.get("kind", "?")
                name     = chunk.get("name") or "(unnamed)"
                score    = chunk.get("score", 0.0)

                print(f"\n  [{i}] {filepath}")
                print(f"        Lines {start}–{end} │ {kind}: {name} │ score: {score:.4f}")

                # Print the text only if show_context is true AND the server provided it
                if self._show_context and "text" in chunk:
                    print()
                    lines = chunk["text"].splitlines()
                    start_idx = start if isinstance(start, int) else 1
                    for lineno, line in enumerate(lines, start=start_idx):
                        print(f"  {lineno:>6} │ {line}")
                    print(f"\n  {THIN_RULE}")
            print()
            
        print(RULE)
        print()

    def run_interactive(self, first_question: str = "") -> None:
        print()
        print(RULE)
        print("  Code Search Client  —  interactive mode")
        print(f"  Server: {self._base_url}")
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
        description="Interactive client for the remote code search server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "question",
        nargs="*",
        help="Optional opening question. Answered first, then prompt appears.",
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        metavar="HOST",
        help=f"Server hostname or IP. Default: {DEFAULT_HOST}",
    )
    p.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        metavar="N",
        help=f"Server port. Default: {DEFAULT_PORT}",
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
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    client = CodeSearchClient(
        host=args.host,
        port=args.port,
        show_sources=args.show_sources,
        show_context=args.show_context,
    )

    if args.no_interactive:
        if not args.question:
            parser.error("--no-interactive requires a question argument.")
        client.query(" ".join(args.question))
    else:
        first = " ".join(args.question) if args.question else ""
        client.run_interactive(first_question=first)


if __name__ == "__main__":
    main()
