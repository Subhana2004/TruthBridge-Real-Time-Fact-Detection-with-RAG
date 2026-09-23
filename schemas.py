"""
schemas.py
==========
Single source of truth for every JSON contract passed between agents.
Agents NEVER pass free-text to each other — only these typed objects
(serialized to/from JSON at the Orchestrator boundary).
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class ClaimDomain(str, Enum):
    HEALTH = "health"
    CLIMATE_ENVIRONMENT = "climate_environment"
    SCIENCE_TECH = "science_tech"
    POLITICS_GOVERNANCE = "politics_governance"
    ECONOMY_FINANCE = "economy_finance"
    EDUCATION = "education"
    HUMAN_RIGHTS_SOCIAL = "human_rights_social"
    HISTORY = "history"
    GENERAL_NEWS_EVENT = "general_news_event"
    OTHER = "other"


class Verdict(str, Enum):
    # TRUE is the public API value.  VERIFIED remains an alias so older
    # clients and stored responses continue to deserialize cleanly.
    TRUE = "TRUE"
    VERIFIED = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    @classmethod
    def _missing_(cls, value):
        # Accept the legacy model spelling without exposing it in responses.
        if isinstance(value, str) and value.upper() == "VERIFIED":
            return cls.TRUE
        return None


# ---------------------------------------------------------------------------
# 1. Claim Detection Agent — output
# ---------------------------------------------------------------------------

class DetectedClaim(BaseModel):
    claim_id: str = Field(..., description="Stable id, e.g. 'c1', 'c2'")
    text: str = Field(..., description="The claim, extracted verbatim or lightly normalized")
    context: Optional[str] = Field(None, description="Surrounding sentence(s) for disambiguation")
    is_checkable: bool = Field(..., description="False if opinion/subjective/unverifiable")
    domain_hint: ClaimDomain = Field(..., description="Best-guess domain from the detector")


class ClaimDetectionOutput(BaseModel):
    source_text_length: int
    claims: List[DetectedClaim]


# ---------------------------------------------------------------------------
# 2. Source Selection Agent — output
# ---------------------------------------------------------------------------

class SelectedSource(BaseModel):
    name: str
    type: str  # "local_rag" | "web_search" | "api"
    domain: ClaimDomain
    priority: int  # lower = higher trust/priority


class SourceSelectionOutput(BaseModel):
    claim_id: str
    domain: ClaimDomain
    sources: List[SelectedSource]


# ---------------------------------------------------------------------------
# 3. Retrieval Agent (RAG) — output
# ---------------------------------------------------------------------------

class EvidenceChunk(BaseModel):
    source_name: str
    source_url: Optional[str] = None
    text: str
    relevance_score: float
    # Chroma metadata is optional at the API boundary to preserve the
    # existing frontend contract while allowing traceable RAG evidence.
    claim_id: Optional[str] = None
    chunk_id: Optional[str] = None
    retrieved_at: Optional[str] = None
    domain: Optional[str] = None


class RetrievalOutput(BaseModel):
    claim_id: str
    evidence: List[EvidenceChunk]


# ---------------------------------------------------------------------------
# 4. Fact Verification Agent — output
# ---------------------------------------------------------------------------

class VerificationResult(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    evidence_used: List[EvidenceChunk]


# ---------------------------------------------------------------------------
# 5. Response Generation Agent — final output returned to the client
# ---------------------------------------------------------------------------

class ClaimResult(BaseModel):
    claim_id: str
    claim_text: str
    domain: ClaimDomain
    verdict: Verdict
    confidence: float
    explanation: str
    citations: List[EvidenceChunk]


class FactCheckResponse(BaseModel):
    total_claims_detected: int
    checkable_claims: int
    opinions_ignored: int
    results: List[ClaimResult]
