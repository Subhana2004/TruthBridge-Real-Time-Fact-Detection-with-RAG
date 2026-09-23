"""
agents/orchestrator.py
======================
LangGraph orchestration for the fact-checking pipeline.

The graph wires the existing typed agents together without changing their
claim detection, source selection, retrieval, verification, or response
generation behavior.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.claim_detection import ClaimDetectionAgent
from agents.fact_verification import FactVerificationAgent
from agents.response_generation import ResponseGenerationAgent
from agents.source_selection import SourceSelectionAgent
from rag.retrieval import RetrievalAgent
from schemas import (
    ClaimResult,
    ClaimDetectionOutput,
    DetectedClaim,
    FactCheckResponse,
    RetrievalOutput,
    SourceSelectionOutput,
    VerificationResult,
)


class VerificationState(TypedDict, total=False):
    text: str
    request_id: str
    detection: ClaimDetectionOutput
    claims: list[DetectedClaim]
    source_selections: dict[str, SourceSelectionOutput]
    retrievals: dict[str, RetrievalOutput]
    verifications: dict[str, VerificationResult]
    results: list[ClaimResult]
    response: FactCheckResponse


class Orchestrator:
    def __init__(
        self,
        claim_detection: ClaimDetectionAgent = None,
        source_selection: SourceSelectionAgent = None,
        retrieval: RetrievalAgent = None,
        fact_verification: FactVerificationAgent = None,
        response_generation: ResponseGenerationAgent = None,
    ):
        self.claim_detection = claim_detection or ClaimDetectionAgent()
        self.source_selection = source_selection or SourceSelectionAgent()
        self.retrieval = retrieval or RetrievalAgent()
        self.fact_verification = fact_verification or FactVerificationAgent()
        self.response_generation = response_generation or ResponseGenerationAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(VerificationState)
        graph.add_node("detect_claims", self._detect_claims)
        graph.add_node("select_sources", self._select_sources)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("verify_facts", self._verify_facts)
        graph.add_node("generate_responses", self._generate_responses)
        graph.add_node("assemble_response", self._assemble_response)
        graph.add_edge(START, "detect_claims")
        graph.add_edge("detect_claims", "select_sources")
        graph.add_edge("select_sources", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "verify_facts")
        graph.add_edge("verify_facts", "generate_responses")
        graph.add_edge("generate_responses", "assemble_response")
        graph.add_edge("assemble_response", END)
        return graph.compile()

    async def process(self, text: str) -> FactCheckResponse:
        state = await self.graph.ainvoke(
            {"text": text, "request_id": uuid.uuid4().hex}
        )
        return state["response"]

    async def _detect_claims(self, state: VerificationState) -> dict[str, Any]:
        detection = await asyncio.to_thread(
            self.claim_detection.detect, state["text"]
        )
        return {"detection": detection, "claims": detection.claims}

    async def _select_sources(self, state: VerificationState) -> dict[str, Any]:
        selections = await asyncio.gather(
            *(
                asyncio.to_thread(self.source_selection.select, claim)
                for claim in state.get("claims", [])
            )
        )
        return {
            "source_selections": {
                selection.claim_id: selection for selection in selections
            }
        }

    async def _retrieve_evidence(self, state: VerificationState) -> dict[str, Any]:
        async def retrieve(claim: DetectedClaim):
            selection = state["source_selections"][claim.claim_id]
            try:
                return await asyncio.to_thread(
                    self.retrieval.retrieve,
                    claim,
                    selection,
                    state["request_id"],
                )
            except TypeError:
                return await asyncio.to_thread(
                    self.retrieval.retrieve,
                    claim,
                    selection,
                )

        retrievals = await asyncio.gather(
            *(retrieve(claim) for claim in state.get("claims", []))
        )
        return {
            "retrievals": {
                retrieval.claim_id: retrieval for retrieval in retrievals
            }
        }

    async def _verify_facts(self, state: VerificationState) -> dict[str, Any]:
        async def verify(claim: DetectedClaim):
            retrieval = state["retrievals"][claim.claim_id]
            return await asyncio.to_thread(
                self.fact_verification.verify, claim, retrieval
            )

        verifications = await asyncio.gather(
            *(verify(claim) for claim in state.get("claims", []))
        )
        return {
            "verifications": {
                verification.claim_id: verification
                for verification in verifications
            }
        }

    async def _generate_responses(self, state: VerificationState) -> dict[str, Any]:
        async def generate(claim: DetectedClaim):
            verification = state["verifications"][claim.claim_id]
            return await asyncio.to_thread(
                self.response_generation.generate, claim, verification
            )

        results = await asyncio.gather(
            *(generate(claim) for claim in state.get("claims", []))
        )
        return {"results": list(results)}

    async def _assemble_response(self, state: VerificationState) -> dict[str, Any]:
        claims = state.get("claims", [])
        return {
            "response": FactCheckResponse(
                total_claims_detected=len(claims),
                checkable_claims=len(claims),
                opinions_ignored=0,
                results=state.get("results", []),
            )
        }
