"""
agents/fact_verification.py
=============================
Agent 4 — Fact Verification Agent

Single responsibility: given a claim + retrieved evidence, judge the claim
strictly against that evidence and output one of:
  TRUE | FALSE | INSUFFICIENT_EVIDENCE
The verdict enum is enforced by schemas.py, so the model cannot emit
anything outside that set without failing validation (handled gracefully
below as INSUFFICIENT_EVIDENCE).
"""

import logging
import re

from schemas import VerificationResult, Verdict, DetectedClaim, RetrievalOutput
from agents.test_huggingface import verify_claim as verify_with_huggingface

SYSTEM_PROMPT = """You are the Fact Verification Agent inside TruthBridge.
You judge a single claim strictly against the evidence provided. You must NOT
use outside knowledge beyond the evidence given — if the evidence is
insufficient to judge the claim, say so honestly.

Verdicts (choose exactly one):
- TRUE: the evidence clearly supports the claim as true.
- FALSE: the evidence clearly contradicts the claim.
- INSUFFICIENT_EVIDENCE: the provided evidence does not clearly confirm or
  deny the claim.

Respond ONLY in English. Return ONLY valid JSON, no prose, no markdown fences,
matching:
{
  "verdict": "TRUE",
  "confidence": 0.0,
  "reasoning": "1-3 sentence explanation grounded only in the evidence given"
}
confidence is a float between 0 and 1 reflecting how strongly the evidence
supports your verdict (not how important the claim is).
"""


class FactVerificationAgent:
    def __init__(self, llm_client=None, hf_verifier=None):
        # Keep the legacy argument for constructor compatibility; production
        # verification is Hugging Face-only.
        self.hf_verifier = hf_verifier or verify_with_huggingface

    def verify(self, claim: DetectedClaim, retrieval: RetrievalOutput) -> VerificationResult:
        if not retrieval.evidence:
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                reasoning="No evidence was retrieved from trusted sources for this claim.",
                evidence_used=[],
            )

        try:
            logging.getLogger(__name__).info(
                "[VERIFY] Sending claim + %d evidence passages to Hugging Face",
                len(retrieval.evidence),
            )
            raw = self.hf_verifier(
                claim.text,
                [f"{e.source_name}: {e.text}" for e in retrieval.evidence],
            )
        except Exception:
            raw = {}
        try:
            raw = raw if isinstance(raw, dict) else {}
            value = str(raw.get("verdict", "INSUFFICIENT_EVIDENCE")).upper()
            if value == "VERIFIED":
                value = "TRUE"
            if value == "MISLEADING":
                value = "INSUFFICIENT_EVIDENCE"
            verdict = Verdict(value)
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            reasoning = str(raw.get("reasoning") or raw.get("explanation") or "")
            if re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", reasoning):
                reasoning = {
                    Verdict.TRUE: "The provided evidence supports the claim.",
                    Verdict.FALSE: "The provided evidence contradicts the claim.",
                    Verdict.INSUFFICIENT_EVIDENCE: "The provided evidence is insufficient to determine the claim.",
                }[verdict]
            logging.getLogger(__name__).info(
                "[VERIFY] Verdict: %s", verdict.value
            )
        except (ValueError, TypeError, AttributeError):
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            confidence = 0.0
            reasoning = "Verification agent could not produce a reliable judgment for this claim."

        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_used=retrieval.evidence,
        )
