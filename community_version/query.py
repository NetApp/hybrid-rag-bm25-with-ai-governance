#!/usr/bin/env python3
"""
Hybrid RAG query runner (no LangChain, no LLM-driven policy).

Separation of concerns (explicit + auditable):
- BM25 = grounding/evidence channel (entity-biased lexical search)
- Vector kNN = semantic/support channel (phrasing/terminology), typically filtered
  to BM25-anchored documents to prevent semantic drift.

Generation:
- Default: 2-pass LLM
  1) grounded draft from BM25-only evidence (citations [B#])
  2) optional rewrite for clarity using vector context (citations [V#] allowed
     only for non-factual clarifications; factual claims must stay grounded)
- Optional: single-pass with both contexts in separate blocks.

All OpenSearch queries and retrieved hits can be printed (--observability) and/or
saved as JSONL (--save-results) for auditability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flask import Flask, jsonify, request
from openai import OpenAI

from common.config import load_settings
from common.bm25 import bm25_retrieve_chunks, bm25_retrieve_doc_anchors
from common.embeddings import vector_retrieve_chunks
from common.llm import (
    build_grounding_prompt,
    build_refine_prompt,
    build_vector_only_prompt,
    call_llm_chat,
    load_llm,
)
from common.logging import get_logger
from common.models import RetrievalHit
from common.named_entity import extract_entities
from common.opensearch_client import create_hot_client, create_long_client, create_vector_client, MyOpenSearch

LOGGER = get_logger(__name__)

# Citations are expected to be inserted inline in the answer as tags like:
#   [B1]  (grounding chunk #1)
#   [V2]  (vector chunk #2)
# Models sometimes emit grouped citations like "[B1, B2]".
# We therefore extract *tokens* inside any bracket/paren groups.
_BRACKET_GROUP_RE = re.compile(r"\[([^\]]+)\]")
_PAREN_GROUP_RE = re.compile(r"\(([^\)]+)\)")
_CITATION_TOKEN_RE = re.compile(r"\b([BV]\d+)\b")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid RAG query (BM25 grounding + vector semantic support) with full auditability."
    )
    p.add_argument("--question", help="User question to answer.")

    p.add_argument("--observability", action="store_true", default=False)
    p.add_argument("--save-results", type=str, default=None, help="Append JSONL records to this path.")

    # Retrieval knobs
    p.add_argument("--top-k", type=int, default=20, help="Total evidence chunks budget.")
    p.add_argument("--bm25-k", type=int, default=None, help="BM25 chunk budget (default ~60% of top-k).")
    p.add_argument("--vec-k", type=int, default=None, help="Vector chunk budget (default remainder).")
    p.add_argument("--bm25-doc-k", type=int, default=10, help="Doc-level BM25 anchors to fetch.")
    p.add_argument("--neighbor-window", type=int, default=0, help="Add ±N adjacent chunks around BM25 hits.")
    p.add_argument("--vec-filter", choices=["anchor", "none"], default="anchor",
                   help="Filter vector search to BM25-anchored docs when possible.")

    # LLM knobs
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)

    # Service mode
    p.add_argument(
        "--service",
        action="store_true",
        default=False,
        help="Start an OpenAI-compatible REST service instead of running the CLI.",
    )
    p.add_argument("--service-host", type=str, default=None, help="Host to bind the REST service.")
    p.add_argument("--service-port", type=int, default=None, help="Port to bind the REST service.")

    return p.parse_args(argv)


def _extract_citations(answer: str) -> List[str]:
    """Extract citation tokens from an LLM answer.

    We expect citations like ``[B1]`` / ``[V2]``.
    Some models emit grouped citations such as ``[B1, B2]`` or ``(B1)``.
    This extractor therefore:
      1) finds bracketed groups ``[...]`` and parenthesized groups ``(...)``
      2) extracts citation *tokens* (B/V + digits) within those groups
    """

    if not answer:
        return []

    cites: set[str] = set()

    for grp in _BRACKET_GROUP_RE.findall(answer):
        for tok in _CITATION_TOKEN_RE.findall(grp):
            cites.add(tok)

    # Parentheses are more ambiguous, so only consider groups that look like citations.
    for grp in _PAREN_GROUP_RE.findall(answer):
        if "B" not in grp and "V" not in grp:
            continue
        for tok in _CITATION_TOKEN_RE.findall(grp):
            cites.add(tok)

    def _key(tag: str) -> Tuple[int, int, str]:
        # Sort B before V, then numeric id.
        prefix = tag[:1]
        num = 10**9
        try:
            num = int(tag[1:])
        except Exception:
            pass
        return (0 if prefix == "B" else 1, num, tag)

    return sorted(cites, key=_key)


def _normalize_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Validate and normalize chat messages from a REST payload.

    Args:
        payload: Parsed JSON payload.
    Returns:
        Normalized list of role/content dicts.
    Raises:
        ValueError: When the payload is missing or malformed.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected non-empty 'messages' list.")
    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a JSON object.")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message requires string 'role' and 'content'.")
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_prompt(messages: List[Dict[str, str]]) -> str:
    """Extract the user prompt from a list of chat messages.

    Args:
        messages: Normalized chat messages.
    Returns:
        The latest user message content, or the last message if no user role exists.
    Raises:
        ValueError: If the message list is empty.
    """
    if not messages:
        raise ValueError("No messages provided.")
    for item in reversed(messages):
        if item.get("role") == "user" and item.get("content"):
            return item["content"]
    return messages[-1]["content"]


def _build_chat_response(*, model: str, content: str) -> Dict[str, Any]:
    """Format a response payload to match the OpenAI chat completion schema."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _resolve_service_bindings(args: argparse.Namespace) -> Tuple[str, int]:
    """Resolve host/port settings for the REST service."""
    host = args.service_host or os.getenv("RAG_AGENT_HOST", "0.0.0.0")
    port = args.service_port or int(os.getenv("RAG_AGENT_PORT", "8002"))
    return host, port


def _error(status: int, message: str) -> tuple[Dict[str, Any], int]:
    """Return a JSON API error payload."""
    return {"error": {"message": message, "type": "invalid_request_error"}}, status


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_one(
    question: str,
    *,
    bm25_hot_client: MyOpenSearch,
    bm25_long_client: MyOpenSearch,
    vec_client: MyOpenSearch,
    llm: OpenAI,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Any]]:
    entities = extract_entities(question)

    settings = load_settings()

    # Indices
    bm25_full_index = settings.opensearch_full_index
    bm25_hot_chunk_index = settings.opensearch_hot_index
    bm25_long_chunk_index = settings.opensearch_long_index
    vec_index = settings.opensearch_vector_index

    if not bm25_full_index or not bm25_long_chunk_index or not vec_index:
        raise RuntimeError("Could not resolve required index names (full, chunk, vector). Check settings/CLI overrides.")

    # Budget split
    top_k = int(args.top_k)
    bm25_k = int(args.bm25_k) if args.bm25_k is not None else max(1, int(round(top_k * 0.6)))
    vec_k = int(args.vec_k) if args.vec_k is not None else max(0, top_k - bm25_k)
    if bm25_k + vec_k != top_k:
        vec_k = max(0, top_k - bm25_k)

    # 1) doc anchors
    anchor_paths, bm25_doc_query, bm25_doc_raw = bm25_retrieve_doc_anchors(
        bm25_long_client,
        bm25_full_index,
        question=question,
        entities=entities,
        k=int(args.bm25_doc_k),
        observability=args.observability,
    )

    # 2A) bm25 HOT INDEX chunks
    bm25_hot_hits, bm25_hot_chunk_query, bm25_hot_chunk_raw = bm25_retrieve_chunks(
        bm25_hot_client,
        bm25_hot_chunk_index,
        question=question,
        entities=entities,
        k=bm25_k,
        anchor_paths=anchor_paths if anchor_paths else None,
        neighbor_window=int(args.neighbor_window),
        observability=args.observability,
    )

    # 2B) bm25 LONG INDEX which contains users' personal data
    bm25_long_hits, bm25_long_chunk_query, bm25_long_chunk_raw = bm25_retrieve_chunks(
        bm25_long_client,
        bm25_long_chunk_index,
        question=question,
        entities=entities,
        k=bm25_k,
        anchor_paths=anchor_paths if anchor_paths else None,
        neighbor_window=int(args.neighbor_window),
        observability=args.observability,
    )

    # Combine BM25 hits from both indices, deduplicating
    seen_bm25: set[Tuple[str, int]] = set()
    combined_bm25_hits: List[RetrievalHit] = []
    for h in bm25_hot_hits + bm25_long_hits:
        key = (h.path, h.chunk_index)
        if key in seen_bm25:
            continue
        combined_bm25_hits.append(h)
        seen_bm25.add(key)
        if len(combined_bm25_hits) >= bm25_k:
            break

    # 3) vector chunks
    vec_anchor_paths: Optional[List[str]] = None
    if args.vec_filter == "anchor" and len(anchor_paths) >= 2:
        vec_anchor_paths = anchor_paths

    vec_hits: List[RetrievalHit] = []
    vec_query: Dict[str, Any] = {}
    vec_raw: List[Dict[str, Any]] = []
    if vec_k > 0:
        vec_hits, vec_query, vec_raw = vector_retrieve_chunks(
            vec_client,
            vec_index,
            question=question,
            anchor_paths=vec_anchor_paths,
            k=vec_k,
            candidate_k=max(vec_k * 5, 50),
            observability=args.observability,
            vector_field="embedding",
        )
        # deterministic top-up if anchor filter starves results
        if vec_anchor_paths and len(vec_hits) < vec_k:
            topup, _, _ = vector_retrieve_chunks(
                vec_client,
                vec_index,
                question=question,
                anchor_paths=None,
                k=vec_k * 2,
                candidate_k=max(vec_k * 10, 100),
                observability=args.observability,
                vector_field="embedding",
            )
            seen = {(h.path, h.chunk_index) for h in vec_hits}
            for h in topup:
                key = (h.path, h.chunk_index)
                if key in seen:
                    continue
                vec_hits.append(h)
                seen.add(key)
                if len(vec_hits) >= vec_k:
                    break
            vec_hits = [RetrievalHit(**{**h.to_jsonable(), "handle": f"V{i+1}"}) for i, h in enumerate(vec_hits)]

    if args.observability:
        print("\n[ENTITIES]", entities)
        print(f"\n[ANCHORS] {len(anchor_paths)}")
        for pth in anchor_paths[:10]:
            print("  -", pth)
        print(f"\n[BM25_HOT_HITS] {len(bm25_hot_hits)}")
        for h in bm25_hot_hits[: min(10, len(bm25_hot_hits))]:
            print(f"  {h.handle} score={h.score:.3f} chunk={h.chunk_index} path={h.path}")
        print(f"\n[BM25_LONG_HITS] {len(bm25_long_hits)}")
        for h in bm25_long_hits[: min(10, len(bm25_long_hits))]:
            print(f"  {h.handle} score={h.score:.3f} chunk={h.chunk_index} path={h.path}")
        # print(f"\n[GRAPH_COMBINED_HITS] {len(graph_hits)}")
        # for h in graph_hits[: min(10, len(graph_hits))]:
        #    print(f"  {h.handle} store={h.store} score={h.score:.3f} chunk={h.chunk_index} path={h.path}")
        print(f"\n[VEC_HITS] {len(vec_hits)} filter={'ON' if vec_anchor_paths else 'OFF'}")
        for h in vec_hits[: min(10, len(vec_hits))]:
            print(f"  {h.handle} score={h.score:.3f} chunk={h.chunk_index} path={h.path}")

    # Generation
    model = settings.llm_server_model
    max_tokens = settings.llama_ctx
    temperature = float(args.temperature)
    top_p = float(args.top_p)

    grounded_draft: Optional[str] = None
    if seen_bm25:
        msgs_a = build_grounding_prompt(question, bm25_hits=combined_bm25_hits, observability=args.observability)
    else:
        # no BM25 evidence, use vector-only evidence (still citation-restricted)
        msgs_a = build_vector_only_prompt(question, vec_hits=vec_hits, observability=args.observability)

    # ground the initial draft in BM25 (or vector-only) evidence
    grounded_draft = call_llm_chat(llm, messages=msgs_a, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    if seen_bm25 and vec_hits:
        msgs_b = build_refine_prompt(question, grounded_draft=grounded_draft, vec_hits=vec_hits, observability=args.observability)
        answer = call_llm_chat(llm, messages=msgs_b, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    else:
        answer = grounded_draft

    citations = _extract_citations(answer)

    audit: Dict[str, Any] = {
        "question": question,
        "entities": entities,
        "indices": {
            "bm25_full": bm25_full_index,
            "bm25_hot_chunks": bm25_hot_chunk_index,
            "bm25_long_chunks": bm25_long_chunk_index,
            "vector_chunks": vec_index,
        },
        "retrieval": {
            "anchor_paths": anchor_paths,
            "bm25_doc_query": bm25_doc_query,
            "bm25_hot_chunk_query": bm25_hot_chunk_query,
            "bm25_long_chunk_query": bm25_long_chunk_query,
            "vector_query": vec_query,
            "bm25_doc_hits": [
                {
                    "filepath": (h.get("_source", {}) or {}).get("filepath"),
                    "category": (h.get("_source", {}) or {}).get("category"),
                    "score": float(h.get("_score") or 0.0),
                }
                for h in bm25_doc_raw
            ],
            "bm25_long_hits": [h.to_jsonable() for h in bm25_long_hits],
            "bm25_hot_hits": [h.to_jsonable() for h in bm25_hot_hits],
            "vector_hits": [h.to_jsonable() for h in vec_hits],
        },
        "generation": {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "grounded_draft": grounded_draft,
            "final_answer": answer,
            "citations_in_answer": citations,
        },
    }

    return answer, audit


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_queries(
    questions: List[str],
    *,
    args: argparse.Namespace,
) -> None:
    bm25_hot_client, _ = create_hot_client()
    bm25_long_client, _ = create_long_client()
    vec_client, _ = create_vector_client()
    llm = load_llm()

    for question in questions:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        start = time.time()
        answer, audit = run_one(
            question,
            bm25_hot_client=bm25_hot_client,
            bm25_long_client=bm25_long_client,
            vec_client=vec_client,
            llm=llm,
            args=args,
        )
        elapsed = time.time() - start

        print("\n" + "=" * 100)
        print("ANSWER:")
        print(answer)
        print("\n" + "=" * 100)
        print(f"Query time: {elapsed:.2f}s")

        cites = audit.get("generation", {}).get("citations_in_answer", []) or []
        print("\nCitations used in answer:", ", ".join(cites) if cites else "(none)")

        bad_cites = audit.get("generation", {}).get("citations_invalid_in_answer", []) or []
        if bad_cites:
            print("Invalid/unknown citation tags in answer:", ", ".join(bad_cites))

        if audit.get("generation", {}).get("citation_repair_applied"):
            print("Citation repair pass:", "APPLIED")

        if args.save_results:
            audit["timing_s"] = elapsed
            audit["created_at_ms"] = int(time.time() * 1000)
            append_jsonl(args.save_results, audit)
            print(f"\nSaved JSONL record to: {args.save_results}")


def create_service_app(args: argparse.Namespace) -> Flask:
    """Create a Flask app that serves the Hybrid RAG pipeline via OpenAI-compatible APIs."""
    app = Flask(__name__)
    settings = load_settings()

    bm25_hot_client, _ = create_hot_client()
    bm25_long_client, _ = create_long_client()
    vec_client, _ = create_vector_client()
    llm = load_llm()

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Dict[str, Any], int]:
        host, port = _resolve_service_bindings(args)
        return jsonify(
            {
                "status": "ok",
                "model": settings.llm_server_model,
                "server": {"host": host, "port": port},
            }
        ), 200

    @app.route("/v1/models", methods=["GET"])
    def models() -> tuple[Dict[str, Any], int]:
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.llm_server_model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "hybrid-rag",
                    }
                ],
            }
        ), 200

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions() -> tuple[Dict[str, Any], int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "Expected JSON object payload.")
        if payload.get("stream") is True:
            return _error(400, "Streaming responses are not supported.")

        try:
            messages = _normalize_messages(payload)
            question = _extract_prompt(messages)
        except ValueError as exc:
            return _error(400, str(exc))

        model = str(payload.get("model") or settings.llm_server_model)
        request_args = argparse.Namespace(**vars(args))
        if "temperature" in payload:
            request_args.temperature = float(payload["temperature"])
        if "top_p" in payload:
            request_args.top_p = float(payload["top_p"])

        answer, _audit = run_one(
            question,
            bm25_hot_client=bm25_hot_client,
            bm25_long_client=bm25_long_client,
            vec_client=vec_client,
            llm=llm,
            args=request_args,
        )

        return jsonify(_build_chat_response(model=model, content=answer)), 200

    return app


def run_service(args: argparse.Namespace) -> None:
    """Run the Hybrid RAG REST service."""
    app = create_service_app(args)
    host, port = _resolve_service_bindings(args)
    app.run(host=host, port=port, debug=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.service:
        run_service(args)
        return

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Keep default examples neutral to avoid injecting unrelated entities.
        questions = [
            "How much did Google purchase Windsurf for?",
            "How much did OpenAI purchase Windsurf for?",
        ]

    run_queries(questions, args=args)


if __name__ == "__main__":
    main()
