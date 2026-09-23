"""
agents/claim_detection.py
==========================
Agent 1 — Claim Detection Agent

Single responsibility: given raw text, extract discrete factual claims,
mark which ones are actually checkable (drop opinions/subjective statements),
and attach a best-guess domain to each. Nothing else — no verification,
no source lookup happens here.
"""

import json
import re
from schemas import ClaimDetectionOutput, DetectedClaim, ClaimDomain
from llm_client import LLMClient

SYSTEM_PROMPT = """You are the Claim Detection Agent inside TruthBridge, a media
literacy pipeline. Your ONLY job: read the input text and extract discrete,
checkable factual claims.

Rules:
- A "claim" is a statement that asserts something as objectively true or false
  (a number, an event, a causal statement, an attribution of a quote, a scientific
  fact, a policy fact, etc).
- Opinions, subjective judgments, predictions of taste, and rhetorical statements
  are NOT claims (e.g. "I think blue is calming" is not a claim).
- Split compound sentences into separate claims when they assert multiple
  distinct facts.
- For each claim, guess its domain from this fixed list only:
  health, climate_environment, science_tech, politics_governance,
  economy_finance, education, human_rights_social, history,
  general_news_event, other

Return ONLY valid JSON, no prose, no markdown fences, matching this exact shape:
{
  "claims": [
    {
      "claim_id": "c1",
      "text": "...",
      "context": "...",
      "is_checkable": true,
      "domain_hint": "health"
    }
  ]
}
If there are no checkable claims, return {"claims": []}.
"""


class ClaimDetectionAgent:
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()

    def detect(self, text: str) -> ClaimDetectionOutput:
        user_prompt = f"Text to analyze:\n\n{text}"
        try:
            raw = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)
        except ValueError:
            # Keep the retrieval pipeline useful without an LLM key.
            return self._fallback_detect(text)

        claims = []
        for i, c in enumerate(raw.get("claims", [])):
            try:
                domain = ClaimDomain(c.get("domain_hint", "other"))
            except ValueError:
                domain = ClaimDomain.OTHER
            claims.append(DetectedClaim(
                claim_id=c.get("claim_id", f"c{i+1}"),
                text=c["text"],
                context=c.get("context"),
                is_checkable=c.get("is_checkable", True),
                domain_hint=domain,
            ))

        # Only checkable claims proceed down the pipeline
        checkable = [c for c in claims if c.is_checkable]
        return ClaimDetectionOutput(source_text_length=len(text), claims=checkable)

    @staticmethod
    def _fallback_detect(text: str) -> ClaimDetectionOutput:
        claims = []
        for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text.strip()), 1):
            sentence = sentence.strip()
            if len(sentence) < 12 or re.match(r"^(i think|in my opinion|i feel)\b", sentence, re.I):
                continue
            lower = sentence.lower()
            domain = ClaimDomain.OTHER
            for candidate, words in {
                ClaimDomain.HEALTH: ("vaccine", "health", "disease", "medical"),
                ClaimDomain.CLIMATE_ENVIRONMENT: ("climate", "temperature", "emission", "warming"),
                ClaimDomain.HISTORY: ("independence", "independent", "war", "century", "historical"),
                ClaimDomain.SCIENCE_TECH: ("science", " nasa ", "technology", "planet"),
                ClaimDomain.POLITICS_GOVERNANCE: ("government", "president", "election", "law"),
            }.items():
                if any(word in lower for word in words):
                    domain = candidate
                    break
            claims.append(DetectedClaim(
                claim_id=f"c{index}", text=sentence, context=None,
                is_checkable=True, domain_hint=domain,
            ))
        return ClaimDetectionOutput(source_text_length=len(text), claims=claims)
