#!/usr/bin/env python3
"""OpenAI client that calls the Hybrid RAG agent service."""
from __future__ import annotations

import argparse
import os

from openai import OpenAI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the client helper."""
    parser = argparse.ArgumentParser(description="Call the Hybrid RAG OpenAI-compatible service.")
    parser.add_argument("--question", required=True, help="Question to send to the agent service.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAG_AGENT_BASE_URL", "http://localhost:8002/v1"),
        help="Base URL for the agent service (default: http://localhost:8002/v1).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RAG_AGENT_MODEL", "local-llm"),
        help="Model name to send in the request payload.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "local-agent"),
        help="API key placeholder for OpenAI client compatibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Send a chat completion request to the agent service and print the answer."""
    args = parse_args(argv)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.question}],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    print(content)


if __name__ == "__main__":
    main()
