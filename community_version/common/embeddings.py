"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import platform
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import Settings, load_settings
from .models import RetrievalHit


_SUPPORTED_TORCH_DTYPES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class RuntimeEmbeddingOptions:
    """Resolved runtime options for embedding model loading."""

    device: str
    attn_implementation: Optional[str] = None
    torch_dtype_name: Optional[str] = None
    padding_side: Optional[str] = None

    @property
    def torch_dtype(self) -> Optional[torch.dtype]:
        """Return the resolved torch dtype object, if configured."""

        return _resolve_torch_dtype(self.torch_dtype_name)

    def model_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for the HF model loader."""

        kwargs: Dict[str, Any] = {}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        return kwargs

    def tokenizer_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for the tokenizer loader."""

        kwargs: Dict[str, Any] = {}
        if self.padding_side:
            kwargs["padding_side"] = self.padding_side
        return kwargs


def _is_macos() -> bool:
    """Return True when the process is running on macOS."""

    return platform.system() == "Darwin"


def _is_qwen3_embedding(model_name: str) -> bool:
    """Return True when the configured model looks like a Qwen3 embedder."""

    return "qwen3-embedding" in (model_name or "").strip().lower()


def _default_device() -> str:
    """Pick the default embedding device for the current runtime."""

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _normalize_device_name(device: str) -> str:
    """Normalize a device string to its family name."""

    return (device or "").strip().lower().split(":", 1)[0]


def _resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """Resolve and validate a torch dtype override."""

    if dtype_name is None:
        return None

    normalized = dtype_name.strip().lower()
    if not normalized:
        return None
    if normalized not in _SUPPORTED_TORCH_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_TORCH_DTYPES))
        raise ValueError(
            f"Unsupported EMBEDDING_TORCH_DTYPE={dtype_name!r}. "
            f"Supported values: {supported}."
        )
    return _SUPPORTED_TORCH_DTYPES[normalized]


def _resolve_runtime_options(model_name: str) -> RuntimeEmbeddingOptions:
    """Resolve device and model-loading options for the embedding runtime."""

    env_device = os.getenv("EMBEDDING_DEVICE", "").strip()
    device = env_device or _default_device()

    is_qwen3 = _is_qwen3_embedding(model_name)
    is_macos = _is_macos()
    device_family = _normalize_device_name(device)

    env_attn = os.getenv("EMBEDDING_ATTN_IMPLEMENTATION", "").strip()
    env_dtype = os.getenv("EMBEDDING_TORCH_DTYPE", "").strip()
    env_padding = os.getenv("EMBEDDING_PADDING_SIDE", "").strip()

    attn_implementation: Optional[str] = env_attn or None
    torch_dtype_name: Optional[str] = env_dtype or None
    padding_side: Optional[str] = env_padding.lower() if env_padding else None

    if is_qwen3:
        if padding_side is None:
            padding_side = "left"
        if is_macos:
            if attn_implementation is None:
                attn_implementation = "eager"
            if torch_dtype_name is None:
                torch_dtype_name = "float16" if device_family == "mps" else "float32"

    # Validate overrides and defaults up front so errors happen during model load.
    _resolve_torch_dtype(torch_dtype_name)

    return RuntimeEmbeddingOptions(
        device=device,
        attn_implementation=attn_implementation,
        torch_dtype_name=torch_dtype_name,
        padding_side=padding_side,
    )


def _instantiate_sentence_transformer(
    model_name: str,
    *,
    device: str,
    model_kwargs: Dict[str, Any],
    tokenizer_kwargs: Dict[str, Any],
) -> SentenceTransformer:
    """Instantiate SentenceTransformer with backward-compatible kwargs."""

    kwargs: Dict[str, Any] = {
        "device": device,
        "model_kwargs": model_kwargs,
    }
    if tokenizer_kwargs:
        kwargs["tokenizer_kwargs"] = tokenizer_kwargs

    try:
        return SentenceTransformer(model_name, **kwargs)
    except TypeError as exc:
        # Some sentence-transformers releases expose `processor_kwargs`
        # instead of `tokenizer_kwargs`. Fall back without changing callers.
        if "tokenizer_kwargs" not in kwargs or "tokenizer_kwargs" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["processor_kwargs"] = fallback_kwargs.pop("tokenizer_kwargs")
        return SentenceTransformer(model_name, **fallback_kwargs)


def _apply_runtime_tokenizer_options(
    model: SentenceTransformer,
    runtime_options: RuntimeEmbeddingOptions,
) -> None:
    """Apply post-load tokenizer safety options when available."""

    if not runtime_options.padding_side:
        return

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return

    try:
        tokenizer.padding_side = runtime_options.padding_side
    except Exception:
        # Some tokenizers may not expose this attribute mutably. In that case,
        # the constructor kwargs above were our best effort.
        pass


@lru_cache(maxsize=8)
def _load_model(
    model_name: str,
    runtime_options: RuntimeEmbeddingOptions,
) -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model."""

    model = _instantiate_sentence_transformer(
        model_name,
        device=runtime_options.device,
        model_kwargs=runtime_options.model_kwargs(),
        tokenizer_kwargs=runtime_options.tokenizer_kwargs(),
    )
    _apply_runtime_tokenizer_options(model, runtime_options)
    return model


@lru_cache(maxsize=1)
def load_embedder() -> "EmbeddingModel":
    """Return a cached :class:`EmbeddingModel` instance."""

    return EmbeddingModel()


@dataclass
class EmbeddingModel:
    """Wrapper around a sentence-transformers embedding model with lazy init."""

    settings: Settings
    _cached_model: Optional[SentenceTransformer] = None
    _runtime_options: Optional[RuntimeEmbeddingOptions] = None

    def __init__(self, settings: Optional[Settings] = None) -> None:
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self._cached_model = None
        self._runtime_options = None

    @property
    def model_name(self) -> str:
        """Determine the configured model name with backward compatibility."""

        name = getattr(self.settings, "embedding_model_name", None)
        if not name:
            name = getattr(self.settings, "embedding_model", None)
        if not name:
            name = "thenlper/gte-small"
        return name

    @property
    def runtime_options(self) -> RuntimeEmbeddingOptions:
        """Resolve model-loading options for the active runtime."""

        if self._runtime_options is None:
            self._runtime_options = _resolve_runtime_options(self.model_name)
        return self._runtime_options

    @property
    def model(self) -> SentenceTransformer:
        """Load the underlying embedding model on first access."""

        if self._cached_model is None:
            self._cached_model = _load_model(self.model_name, self.runtime_options)
        return self._cached_model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, respecting explicit overrides."""

        dim = getattr(self.settings, "embedding_dimension", None)
        if dim is not None:
            return int(dim)

        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            vec = self.encode(["_probe_"])[0]
            return int(vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """Encode text to L2-normalized numpy arrays ready for cosine kNN."""

        items: List[str] = list(texts)
        if len(items) == 0:
            return []

        arr = self.model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(arr.shape[0])]
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(items) == 1:
            return [arr]
        return [np.asarray(v, dtype=float) for v in arr]


def embed_question(question: str) -> List[float]:
    """Embed a single question into a plain Python list."""

    embedder = load_embedder()
    vecs = embedder.encode([question])
    return to_list(vecs[0])


def normalize_vector_hits(
    res_vector: Dict[str, Any], *, label: str = "VECTOR"
) -> List[Dict[str, Any]]:
    """Normalize vector hits to look like BM25 hits for downstream reuse."""

    hits = res_vector.get("hits", {}).get("hits", []) or []
    out: List[Dict[str, Any]] = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        hit["_store_label"] = res_vector.get("_store_label", label)
        hit["_index_used"] = res_vector.get("_index_used", "?")
        source = hit.get("_source") or {}
        if not isinstance(source, dict):
            source = {}
        if "filepath" not in source:
            source["filepath"] = source.get("path") or source.get("rel_path") or "<unknown>"
        if "content" not in source:
            source["content"] = source.get("text") or ""
        hit["_source"] = source
        out.append(hit)

    return out


def to_list(vec: Union[np.ndarray, Sequence[float], List[float]]) -> List[float]:
    """Convert a vector-like object to a Python list of floats."""

    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]


def _build_vector_query(
    query_vector: List[float],
    *,
    k: int,
    candidate_k: int,
    anchor_paths: Optional[List[str]] = None,
    vector_field: str = "embedding",
) -> Dict[str, Any]:
    knn_clause = {"knn": {vector_field: {"vector": query_vector, "k": candidate_k}}}
    return {
        "size": k,
        "_source": ["path", "category", "chunk_index", "chunk_count", "text"],
        "query": {
            "bool": {
                "must": [knn_clause],
                "filter": [{"terms": {"path": anchor_paths}}] if anchor_paths else [],
            }
        },
    }


def vector_retrieve_chunks(
    vec_client: Any,
    index: str,
    *,
    question: str,
    anchor_paths: Optional[List[str]],
    k: int,
    candidate_k: int,
    observability: bool,
    vector_field: str = "embedding",
) -> Tuple[List[RetrievalHit], Dict[str, Any], List[Dict[str, Any]]]:
    embedder = EmbeddingModel()
    qvec = to_list(embedder.encode([question])[0])

    q = _build_vector_query(qvec, k=k, candidate_k=candidate_k, anchor_paths=anchor_paths, vector_field=vector_field)

    if observability:
        dbg = json.loads(json.dumps(q))
        try:
            dbg["query"]["bool"]["must"][0]["knn"][vector_field]["vector"] = "<omitted>"
        except Exception:
            pass
        print("\n[VECTOR_QUERY]\n" + json.dumps(dbg, ensure_ascii=False, indent=2, sort_keys=True, default=str))

    res = vec_client.search(index=index, body=q)
    raw_hits = res.get("hits", {}).get("hits", []) or []

    hits: List[RetrievalHit] = []
    for i, h in enumerate(raw_hits, start=1):
        src = h.get("_source", {}) or {}
        hits.append(
            RetrievalHit(
                channel="vector",
                handle=f"V{i}",
                index=index,
                os_id=h.get("_id", ""),
                score=float(h.get("_score") or 0.0),
                path=src.get("path") or "",
                category=src.get("category") or "",
                chunk_index=src.get("chunk_index"),
                chunk_count=src.get("chunk_count"),
                text=(src.get("text") or "").strip(),
            )
        )

    return hits, q, raw_hits


__all__ = [
    "load_embedder",
    "EmbeddingModel",
    "embed_question",
    "normalize_vector_hits",
    "_build_vector_query",
    "vector_retrieve_chunks",
    "to_list",
]
