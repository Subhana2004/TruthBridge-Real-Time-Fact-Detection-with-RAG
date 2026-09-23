"""
agents/source_selection.py
============================
Agent 2 — Source Selection Agent

Single responsibility: given a claim + its domain, select which trusted
sources should be consulted. This is deliberately rule-based (a lookup
against trusted_sources.py) rather than LLM-based — trust decisions should
be deterministic and auditable, not a model's guess.
"""

from schemas import SourceSelectionOutput, SelectedSource, DetectedClaim
from trusted_sources import get_sources_for_domain


class SourceSelectionAgent:
    def select(self, claim: DetectedClaim) -> SourceSelectionOutput:
        raw_sources = get_sources_for_domain(claim.domain_hint)
        sources = [
            SelectedSource(
                name=s["name"],
                type=s["type"],
                domain=claim.domain_hint,
                priority=s["priority"],
            )
            for s in raw_sources
        ]
        sources.sort(key=lambda s: s.priority)
        return SourceSelectionOutput(
            claim_id=claim.claim_id,
            domain=claim.domain_hint,
            sources=sources,
        )
