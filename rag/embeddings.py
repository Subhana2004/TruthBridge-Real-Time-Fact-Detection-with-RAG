"""Local sentence-transformer embeddings used by the live RAG pipeline.

The model is loaded lazily so importing the API never requires a network call.
If the optional model cannot be loaded, a small deterministic token embedding
is used as a safe degradation for development and offline tests.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from functools import lru_cache
from typing import Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def _model():
    # Current Windows wheels for torch/sentence-transformers can terminate
    # CPython 3.14 while loading native DLLs.  Do not import them on that
    # runtime unless an operator explicitly opts in.
    if sys.version_info >= (3, 14) and os.getenv("TRUTHBRIDGE_FORCE_LOCAL_EMBEDDINGS") != "1":
        LOGGER.warning("Local model disabled on Python %s; use a supported torch wheel", sys.version.split()[0])
        return None
    try:
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading local embedding model: %s", MODEL_NAME)
        return SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # pragma: no cover - depends on local install/model cache
        LOGGER.warning("Local sentence-transformer unavailable; using fallback: %s", exc)
        return None


def _fallback(text: str) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    for token in re.findall(r"\b\w+\b", text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        vector[int.from_bytes(digest, "big") % EMBEDDING_DIMENSION] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    values = [str(text or "") for text in texts]
    if not values:
        return []
    LOGGER.info("[EMBED] Creating embeddings for %d chunks", len(values))
    model = _model()
    if model is not None:
        vectors = model.encode(
            values,
            batch_size=min(32, len(values)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"{MODEL_NAME} returned {getattr(vectors, 'shape', None)}; "
                f"expected {EMBEDDING_DIMENSION} dimensions"
            )
        return vectors.tolist()
    return [_fallback(value).tolist() for value in values]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
