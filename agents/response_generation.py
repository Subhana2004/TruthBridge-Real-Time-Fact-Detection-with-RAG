"""
agents/response_generation.py
===============================
Agent 5 — Response Generation Agent

Single responsibility: turn a VerificationResult into a clear, user-facing
explanation with citations. Does not re-judge the claim — the verdict is
already final by the time this agent runs; it only handles presentation.
"""

from schemas import ClaimResult, DetectedClaim, VerificationResult
from llm_client import LLMClient

SYSTEM_PROMPT = """You are the Response Generation Agent inside TruthBridge.
You are given a claim, its verdict, and the reasoning behind it. Write a
short, plain-language explanation (2-4 sentences) a general audience can
understand, suitable for display directly under the claim in a browser
extension. Do not change the verdict. Do not invent new evidence.

Return ONLY valid JSON, no prose, no markdown fences:
{ "explanation": "..." }
"""


class ResponseGenerationAgent:
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()

    def generate(self, claim: DetectedClaim, verification: VerificationResult) -> ClaimResult:
        user_prompt = f"""Claim: "{claim.text}"
Verdict: {verification.verdict.value}
Reasoning: {verification.reasoning}"""

        try:
            raw = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)
            explanation = raw.get("explanation", verification.reasoning)
        except ValueError:
            explanation = verification.reasoning

        return ClaimResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            domain=claim.domain_hint,
            verdict=verification.verdict,
            confidence=verification.confidence,
            explanation=explanation,
            citations=verification.evidence_used,
        )
