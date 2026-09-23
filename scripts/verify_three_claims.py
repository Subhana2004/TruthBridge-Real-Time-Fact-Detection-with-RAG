"""Run the live web RAG retriever against three claims.

Usage: python scripts/verify_three_claims.py
The script performs fresh searches and does not use a source URL catalogue.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retrieval import LiveWebSearchRetriever
from agents.fact_verification import FactVerificationAgent
from schemas import DetectedClaim, ClaimDomain, EvidenceChunk, RetrievalOutput


CLAIMS = [
    ("India became independent in 1989.", "FALSE"),
    ("India became independent on 15 August 1947.", "TRUE"),
    ("The 2026 FIFA World Cup will be hosted by Canada, Mexico, and the United States.", None),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    retriever = LiveWebSearchRetriever()
    verifier = FactVerificationAgent()
    for index, (claim_text, expected) in enumerate(CLAIMS, 1):
        print(f"\nCLAIM: {claim_text}")
        evidence = retriever.search(
            claim_text,
            [],
            n_results=3,
            request_id="standalone-rag-test",
            claim_id=f"c{index}",
            domain=ClaimDomain.HISTORY.value if index < 3 else ClaimDomain.GENERAL_NEWS_EVENT.value,
        )
        if not evidence:
            print("  No evidence returned (search/fetch failures are skipped).")
        for item in evidence:
            print(f"  [{item['relevance_score']:.3f}] {item['source_name']} {item['source_url']}")
            print(f"    {item['text']}")
        if evidence:
            retrieval = RetrievalOutput(
                claim_id=f"c{index}",
                evidence=[EvidenceChunk(**item) for item in evidence],
            )
            result = verifier.verify(
                DetectedClaim(
                    claim_id=f"c{index}",
                    text=claim_text,
                    is_checkable=True,
                    domain_hint=ClaimDomain.HISTORY if index < 3 else ClaimDomain.GENERAL_NEWS_EVENT,
                ),
                retrieval,
            )
            expected_text = f" (expected {expected})" if expected else ""
            print(f"  VERDICT: {result.verdict.value} confidence={result.confidence:.2f}{expected_text}")
        else:
            print("  VERDICT: INSUFFICIENT_EVIDENCE (no usable fetched evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
