"""ChromaDB vector storage backed by local all-MiniLM-L6-v2 embeddings.

Persistent collections are used for optional seed data. Live web evidence is
inserted into an ephemeral per-query collection and is never mixed with it.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from rag.embeddings import embed_texts

LOGGER = logging.getLogger(__name__)
VECTOR_DIR = Path(
    os.getenv("TRUTHBRIDGE_CHROMA_DIR")
    or os.getenv("TRUTHBRIDGE_VECTOR_DIR")
    or "./chroma_store"
)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
_client = None
# Chroma is the production current-query store.  An explicit opt-out is useful
# for minimal/offline installations, and every Chroma operation still has a
# safe lexical fallback if the native backend is unavailable.
_USE_CHROMA = os.getenv("TRUTHBRIDGE_USE_CHROMA", "1").lower() not in {"0", "false", "no"}
# Chroma's Rust bindings currently terminate CPython 3.14 on Windows during
# collection.add/query.  Keep Chroma enabled by default on supported runtimes
# and use the non-crashing semantic/lexical fallback on this runtime.
_CHROMA_RUNTIME_SAFE = sys.version_info < (3, 14) and not (
    os.name == "nt" and getattr(chromadb, "__version__", "").startswith("1.5")
)


def _persistent_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    return _client


def _collection(name: str, client=None):
    return (client or _persistent_client()).get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(collection_name: str, docs: list[dict]):
    if not docs or not _USE_CHROMA or not _CHROMA_RUNTIME_SAFE:
        return
    collection = _collection(collection_name)
    texts = [str(doc.get("text", "")) for doc in docs]
    valid = [(doc, text) for doc, text in zip(docs, texts) if text.strip()]
    if not valid:
        return
    documents = [text for _, text in valid]
    collection.upsert(
        ids=[str(doc.get("id") or uuid.uuid4()) for doc, _ in valid],
        documents=documents,
        embeddings=embed_texts(documents),
        metadatas=[
            {
                "source_name": str(doc.get("source_name", "Unknown")),
                "source_url": str(doc.get("source_url") or ""),
                "claim_id": str(doc.get("claim_id") or "seed"),
                "retrieved_at": str(doc.get("retrieved_at") or datetime.now(timezone.utc).isoformat()),
                "chunk_id": str(doc.get("chunk_id") or doc.get("id") or uuid.uuid4()),
                "domain": str(doc.get("domain") or collection_name),
            }
            for doc, _ in valid
        ],
    )
    LOGGER.info("Stored %d documents collection=%s", len(documents), collection_name)


def _format_results(result: dict) -> list[dict]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    output = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        output.append(
            {
                "text": text,
                "source_name": metadata.get("source_name", "Unknown"),
                "source_url": metadata.get("source_url") or None,
                "relevance_score": round(max(0.0, 1.0 - float(distance)), 6),
                "claim_id": metadata.get("claim_id"),
                "chunk_id": metadata.get("chunk_id"),
                "retrieved_at": metadata.get("retrieved_at"),
                "domain": metadata.get("domain"),
            }
        )
    return output


def query(collection_name: str, query_text: str, n_results: int = 4) -> list[dict]:
    """Search a persistent collection; absent collections safely return none."""

    if not _USE_CHROMA or not _CHROMA_RUNTIME_SAFE:
        return []
    try:
        collection = _persistent_client().get_collection(collection_name)
    except Exception:
        return []
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_texts([query_text])[0]],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return _format_results(result)


@contextmanager
def temporary_collection(prefix: str = "current_query"):
    """Yield an ephemeral Chroma collection and clean it up on exit."""

    client = chromadb.EphemeralClient()
    name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    collection = _collection(name, client)
    LOGGER.info("Created temporary evidence collection=%s", name)
    try:
        yield collection
    finally:
        try:
            client.delete_collection(name)
        except Exception:
            LOGGER.debug("Temporary collection cleanup failed", exc_info=True)
        LOGGER.info("Removed temporary evidence collection=%s", name)


def query_temporary_evidence(
    query_text: str,
    docs: list[dict],
    n_results: int = 6,
    *,
    request_id: str | None = None,
    claim_id: str | None = None,
    domain: str | None = None,
) -> list[dict]:
    """Add and semantically query evidence in an isolated Chroma collection.

    Each call owns a temporary collection.  Callers provide request/claim
    identifiers so metadata and collection names cannot cross-contaminate
    concurrent requests.
    """

    if not docs:
        return []
    lexical = sorted(docs, key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True)
    request_id = request_id or uuid.uuid4().hex
    claim_id = claim_id or "claim"
    retrieved_at = datetime.now(timezone.utc).isoformat()
    if not _USE_CHROMA or not _CHROMA_RUNTIME_SAFE:
        LOGGER.warning(
            "[CHROMA] Native backend unavailable; current-query evidence remains isolated in memory"
        )
        return [
            {
                "source_name": item.get("source_name", "Unknown"),
                "source_url": item.get("source_url") or None,
                "text": item.get("text", ""),
                "relevance_score": float(item.get("relevance_score", 0.0)),
                "claim_id": item.get("claim_id") or claim_id,
                "chunk_id": item.get("chunk_id") or f"{claim_id}_{index}",
                "retrieved_at": item.get("retrieved_at") or retrieved_at,
                "domain": domain or item.get("domain"),
            }
            for index, item in enumerate(lexical[:n_results])
        ]
    safe_prefix = f"rq_{request_id[:20]}_{claim_id[:20]}"
    retrieved_at = datetime.now(timezone.utc).isoformat()
    valid = [doc for doc in docs if str(doc.get("text", "")).strip()]
    if not valid:
        return []
    try:
        with temporary_collection(prefix=safe_prefix) as collection:
            chunk_ids = [
                str(doc.get("chunk_id") or doc.get("id") or f"{claim_id}_{index}")
                for index, doc in enumerate(valid)
            ]
            collection.add(
                ids=chunk_ids,
                documents=[str(doc["text"]) for doc in valid],
                embeddings=embed_texts([str(doc["text"]) for doc in valid]),
                metadatas=[
                    {
                        "claim_id": str(doc.get("claim_id") or claim_id),
                        "source_url": str(doc.get("source_url") or ""),
                        "source_name": str(doc.get("source_name", "Unknown")),
                        "retrieved_at": str(doc.get("retrieved_at") or retrieved_at),
                        "chunk_id": chunk_id,
                        "domain": str(doc.get("domain") or domain or "other"),
                        "lexical_score": float(doc.get("relevance_score", 0.0)),
                    }
                    for doc, chunk_id in zip(valid, chunk_ids)
                ],
            )
            LOGGER.info("[CHROMA] Added %d evidence chunks", len(valid))
            result = collection.query(
                query_embeddings=[embed_texts([query_text])[0]],
                n_results=min(n_results, len(valid)),
                include=["documents", "metadatas", "distances"],
            )
            ranked = _format_results(result)
            # Preserve lexical/source score while making semantic similarity
            # the primary signal for current-query evidence.
            for item, source in zip(ranked, result.get("metadatas", [[]])[0]):
                item["relevance_score"] = round(
                    0.7 * item["relevance_score"] + 0.3 * float(source.get("lexical_score", 0.0)),
                    6,
                )
            ranked.sort(key=lambda item: item["relevance_score"], reverse=True)
            LOGGER.info("[CHROMA] Retrieved %d evidence chunks", len(ranked))
            return ranked
    except Exception:
        LOGGER.warning("Chroma current-query ranking failed; using lexical scores", exc_info=True)
        ranked = sorted(docs, key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True)
        return [
            {
                "source_name": item.get("source_name", "Unknown"),
                "source_url": item.get("source_url") or None,
                "text": item.get("text", ""),
                "relevance_score": float(item.get("relevance_score", 0.0)),
                "claim_id": item.get("claim_id") or claim_id,
                "chunk_id": item.get("chunk_id") or f"{claim_id}_{index}",
                "retrieved_at": item.get("retrieved_at") or retrieved_at,
                "domain": domain or item.get("domain"),
            }
            for index, item in enumerate(ranked[:n_results])
        ]
