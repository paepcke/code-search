#!/usr/bin/env python
# ############################################
# -*- coding: utf-8 -*-
# @Author: Andreas Paepcke
# @Date:   2026-03-20 09:31:24
# @Last Modified by:   Andreas Paepcke
# @Last Modified time: 2026-03-20 09:32:54
# #############################################

"""
code_search_client.py  --  Interactive client for the remote code search server.

Connects to a ``code_search_server.py`` instance running on a remote host
(e.g. sextus) and provides the same interactive REPL experience as running
``code_query.py`` locally.  Conversation history is maintained here on the
client and sent with every request.

Usage
-----
    python code_search_client.py                        # connect to sextus:58008
    python code_search_client.py --host 10.0.0.5       # explicit IP
    python code_search_client.py --host sextus.local    # mDNS name
    python code_search_client.py "opening question"     # answer then prompt
    python code_search_client.py --no-interactive "..."  # single-shot

Options
-------
    --host   HOST   Hostname or IP of the server.  Default: sextus.local
    --port   N      Server port.                   Default: 58008
    --top-k  N      Chunks to retrieve.            Default: server default
    --no-interactive   Single-shot mode.
    --code-snippets    Print file paths only (server never sends code text).

Special commands (interactive mode)
------------------------------------
    new  /new       Clear conversation history — start a fresh topic.
    quit exit /q    Exit.

Dependencies
------------
    pip install requests
"""

import argparse
import readline  # noqa: F401  — side-effect: Emacs-style keys in input()
import sys
import textwrap
from pathlib import Path

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
    """HTTP client for the code search server.

    Maintains conversation history locally and sends it with every request
    so the server can produce contextually aware follow-up answers.

    :param host:          Hostname or IP address of the server.
    :param port:          TCP port the server listens on.
    :param show_snippets: When True, print chunk metadata in results.
                          (The server never returns code text, only paths
                          and line numbers, so this controls that display.)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        show_snippets: bool = False,
    ) -> None:
        self._base_url     = f"http://{host}:{port}"
        self._show_snippets = show_snippets
        self._history: list[dict] = []
        self._check_server()

    # ------------------------------------------------------------------
    def _check_server(self) -> None:
        """Verify the server is reachable and print its configuration.

        :raises SystemExit: If the server cannot be reached.
        """
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

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear local conversation history to start a new topic."""
        self._history = []

    # ------------------------------------------------------------------
    def query(self, question: str) -> None:
        """Send *question* to the server and print the result.

        :param question: Natural-language question from the user.
        """
        print("Asking server ...")
        try:
            resp = requests.post(
                f"{self._base_url}/query",
                json={"question": question, "history": self._history},
                timeout=180,      # LLM can be slow on a loaded machine
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

        # Update local history from server response
        self._history = data.get("history", self._history)

        self._print_results(
            question=question,
            chunks=data.get("chunks", []),
            answer=data.get("answer", ""),
        )

    # ------------------------------------------------------------------
    def _print_results(
        self,
        question: str,
        chunks: list[dict],
        answer: str,
    ) -> None:
        """Render one query result to stdout.

        :param question: The user's question.
        :param chunks:   Slim chunk dicts (no text field) from the server.
        :param answer:   LLM explanation string.
        """
        print()
        print(RULE)
        print(f"  QUERY:  {question}")
        print(RULE)

        if not chunks:
            print("\n  No matching chunks found.\n")
        else:
            print(f"\n  {len(chunks)} chunk(s) retrieved:\n")
            for i, chunk in enumerate(chunks, 1):
                filepath = chunk.get("filepath", "?")
                start    = chunk.get("start_line", "?")
                end      = chunk.get("end_line",   "?")
                kind     = chunk.get("kind",        "?")
                name     = chunk.get("name") or "(unnamed)"
                score    = chunk.get("score", 0.0)

                print(f"  [{i}]  {filepath}")
                print(f"        Lines {start}–{end}  │  "
                      f"{kind}: {name}  │  score: {score:.4f}")
                print()
                print(f"  {THIN_RULE}")
                print()

        if answer:
            print(RULE)
            print("  LLM EXPLANATION")
            print(RULE)
            print()
            for line in answer.splitlines():
                if len(line) > 78:
                    print(textwrap.fill(line, width=78,
                                        subsequent_indent="  "))
                else:
                    print(line)
            print()

        print(RULE)
        print()

    # ------------------------------------------------------------------
    def run_interactive(self, first_question: str = "") -> None:
        """Enter the interactive REPL loop.

        :param first_question: Optional opening question answered before
                               the prompt appears (e.g. from the CLI).
        """
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
        "--no-interactive",
        action="store_true",
        help="Single-shot mode: answer one question and exit.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Ask server to skip LLM (not yet implemented server-side; "
             "reserved for future use).",
    )
    return p


def main() -> None:
    """Entry point for the client CLI."""
    parser = _build_parser()
    args   = parser.parse_args()

    client = CodeSearchClient(
        host=args.host,
        port=args.port,
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