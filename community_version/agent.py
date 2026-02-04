#!/usr/bin/env python3
"""Entry point for running the Hybrid RAG OpenAI-compatible service."""
from __future__ import annotations

from query import parse_args, run_service


def main(argv: list[str] | None = None) -> None:
    """Start the Hybrid RAG agent service."""
    args = parse_args(argv)
    args.service = True
    run_service(args)


if __name__ == "__main__":
    main()
