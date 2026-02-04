"""LLM loading and hybrid RAG orchestration utilities."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import Settings, load_settings
from .logging import get_logger
from .models import RetrievalHit

LOGGER = get_logger(__name__)


def load_llm(settings: Optional[Settings] = None) -> OpenAI:
    """Construct and cache an OpenAI-compatible client for the LLM server."""

    if settings is None:
        settings = load_settings()

    LOGGER.info("Connecting to LLM server at %s", settings.llm_server_url)
    client = OpenAI(
        base_url=settings.llm_server_url,
        api_key=settings.llm_server_api_key,
    )
    setattr(client, "default_model", settings.llm_server_model)
    return client


def generate_answer(
    llm: Any,
    question: str,
    model: str,
    context: str,
    *,
    observability: bool = False,
    max_tokens: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> str:
    """Run a chat completion against the LLM using the provided context."""

    if not context.strip():
        return "No supporting documents found."

    system_msg = "Answer using ONLY the provided context below."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\n"

    if observability:
        LOGGER.info("LLM prompt context length=%d chars", len(context))

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]
    return call_llm_chat(
        llm,
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting for non-chat completion interfaces."""
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}:\n{content}".strip())
    return "\n\n".join(parts).strip()


def _extract_llm_text(resp: Any) -> str:
    """Normalize many possible completion response shapes into a plain string."""
    if resp is None:
        return ""

    if isinstance(resp, str):
        return resp.strip()

    # llama_cpp and many other clients return dicts
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                # chat completion style
                msg = c0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "").strip()
                # text completion style
                if "text" in c0:
                    return str(c0.get("text") or "").strip()
                # streaming delta style
                delta = c0.get("delta")
                if isinstance(delta, dict) and "content" in delta:
                    return str(delta.get("content") or "").strip()
        # some wrappers return {"content": "..."}
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"].strip()
        return str(resp).strip()

    # OpenAI v1 python client response objects have .choices
    choices = getattr(resp, "choices", None)
    if choices and isinstance(choices, list):
        c0 = choices[0]
        # chat
        msg = getattr(c0, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content.strip()
        # completion
        text = getattr(c0, "text", None)
        if isinstance(text, str):
            return text.strip()

    # Fallback: stringify
    return str(resp).strip()


def call_llm_chat(
    llm: OpenAI,
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Chat completion using the OpenAI client."""
    resp = llm.chat.completions.create(
        model=model,
        messages=messages,
        # temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return _extract_llm_text(resp)


def _format_hits(hits: List[RetrievalHit], *, title: str) -> str:
    """Render retrieval hits into a tag-delimited context block.

    Each hit is wrapped in an explicit open/close tag so multi-line chunks
    remain unambiguous:

        [B1]
        ...text...
        [/B1]

    This makes it easier for smaller instruction models to reliably reference
    chunks and for downstream parsing/auditing to remain robust.
    """

    if not hits:
        return f"=== {title} ===\n(none)"

    blocks: List[str] = [f"=== {title} ==="]
    for h in hits:
        open_tag = f"[{h.handle}]"
        close_tag = f"[/{h.handle}]"
        # Keep metadata minimal to reduce the chance the model quotes it.
        meta: List[str] = []
        if h.path:
            meta.append(f"path={h.path}")
        if h.chunk_index is not None and h.chunk_count is not None:
            meta.append(f"chunk={h.chunk_index}/{h.chunk_count}")
        if h.category:
            meta.append(f"category={h.category}")
        meta_line = f"META: {', '.join(meta)}" if meta else ""

        text = (h.text or "").strip()
        if meta_line:
            blocks.append(f"{open_tag}\n{meta_line}\n{text}\n{close_tag}".strip())
        else:
            blocks.append(f"{open_tag}\n{text}\n{close_tag}".strip())

    return "\n\n".join(blocks).strip()


def _allowed_citation_tags(hits: List[RetrievalHit]) -> List[str]:
    """Return citation tags (opening tags only) for a set of hits."""

    out: List[str] = []
    for h in hits:
        handle = (h.handle or "").strip()
        if not handle:
            continue
        out.append(f"[{handle}]")
    return out


def build_grounding_prompt(question: str, bm25_hits: List[RetrievalHit], observability: Optional[bool] = False) -> List[Dict[str, str]]:
    if observability: 
        LOGGER.info("Building grounding prompt with %d BM25 hits", len(bm25_hits))

    context = _format_hits(bm25_hits, title="BM25 Grounding Evidence (authoritative facts)")
    allowed = " ".join(_allowed_citation_tags(bm25_hits)) if bm25_hits else "(none)"
    system = (
        "You are a grounded QA assistant. Answer using ONLY the BM25 Grounding Evidence.\n"
        "Evidence chunks are delimited as [B#] ... [/B#].\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY sentence that contains a factual claim, append one or more citation tags.\n"
        "- Only cite BM25 the opening tag only (e.g., [B1]). Never use closing tags like [/B1] in your answer.\n"
        "- Don't mention the term BM25 in the answer.\n"
        "\n"
        "If the evidence does not support the answer, write exactly: I don't know based on the provided evidence.\n"
        "Do not quote the evidence headers/metadata.\n"
        "If evidence conflicts, disclose the conflict.\n"
        "Output ONLY the answer text."
    )
    user = f"QUESTION:\n{question}\n\nGROUNDING_EVIDENCE:\n{context}\n"

    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(prompt_details))
        LOGGER.info("=====================================================")

    return prompt_details


def build_vector_only_prompt(question: str, vec_hits: List[RetrievalHit], observability: Optional[bool] = False) -> List[Dict[str, str]]:
    if observability:
        LOGGER.info("Building vector-only prompt with %d vector hits", len(vec_hits))

    context = _format_hits(vec_hits, title="Vector Evidence (semantic fallback)")
    allowed = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    system = (
        "Answer using ONLY the Vector Evidence.\n"
        "Evidence chunks are delimited as [V#] ... [/V#].\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY sentence that contains a factual claim, append one or more citation tags.\n"
        "- Only cite vector opening tag only (e.g., [V1]). Never use closing tags like [/V1] in your answer.\n"
        "- Never invent citation numbers or use tags not listed above.\n"
        "\n"
        "If evidence does not support the answer, write exactly: I don't know based on the provided evidence.\n"
        "Do not quote the evidence headers/metadata.\n"
        "Output ONLY the answer text."
    )
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{context}\n"
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(prompt_details))
        LOGGER.info("=====================================================")

    return prompt_details


def build_refine_prompt(question: str, grounded_draft: str, vec_hits: List[RetrievalHit], observability: Optional[bool] = False) -> List[Dict[str, str]]:
    if observability:
        LOGGER.info("Building refine prompt with %d vector hits", len(vec_hits))

    vec_context = _format_hits(vec_hits, title="Vector Semantic Context (phrasing/terminology support)")
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    system = (
        "Rewrite the grounded draft for clarity and readability.\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Do NOT add any new factual claims beyond what appears in the grounded draft.\n"
        "- Preserve all existing [B#] citations EXACTLY (do not delete, renumber, merge, or move them).\n"
        "- You MAY add brief non-factual clarifications (definitions, paraphrases) supported by the vector context.\n"
        "\n"
        "VECTOR CITATIONS (optional):\n"
        f"- Allowed vector citation tags: {allowed_v}\n"
        "- After EVERY sentence that contains a factual claim, append one or more citation tags.\n"
        f"- Don't mention the word vector or vectors in the answer.\n"
        "- Only cite vector opening tag only (e.g., [V1]). Never use closing tags like [/V1] in your answer.\n"
        "\n"
        "Output ONLY the rewritten answer text.\n"
        "If there is conflicting evidence, disclose the conflict and data.\n"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"GROUNDED_DRAFT:\n{grounded_draft}\n\n"
        f"{vec_context}\n"
    )
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(prompt_details))
        LOGGER.info("=====================================================")

    return prompt_details


__all__ = [
    "load_llm",
    "generate_answer",
    "ask",
    "call_llm_chat",
    "build_grounding_prompt",
    "build_vector_only_prompt",
    "build_refine_prompt",
]
