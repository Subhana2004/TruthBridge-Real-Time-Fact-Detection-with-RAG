"""Reusable Hugging Face fact-verification adapter.

The filename is retained for compatibility with the original manual smoke
test, but importing this module performs no network request and never prints
or returns the configured token.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

MODEL_NAME = os.getenv("TRUTHBRIDGE_HF_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")
NON_ENGLISH_MARKERS = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _parse_json(text: str) -> dict:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Hugging Face returned non-JSON output")
        return json.loads(match.group(0))


def verify_claim(
    claim: str,
    evidence: str | Iterable[str],
    *,
    model: str = MODEL_NAME,
    client=None,
) -> dict:
    """Verify one claim against supplied evidence using Hugging Face.

    A missing token or provider failure raises an exception so the production
    agent can return INSUFFICIENT_EVIDENCE rather than inventing a verdict.
    """

    token = os.getenv("HF_TOKEN")
    if not token and client is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        token = os.getenv("HF_TOKEN")
    if client is None:
        if not token:
            raise ValueError("HF_TOKEN is not configured")
        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=token)
    if isinstance(evidence, str):
        evidence_text = evidence
    else:
        evidence_text = "\n\n".join(str(item) for item in evidence)
    prompt = f"""You are the fact verification engine for TruthBridge.
Evaluate the CLAIM only using the provided EVIDENCE. Do not use outside
knowledge. Respond ONLY in English. The verdict labels, confidence, and
explanation must be in English. Preserve all evidence text and source URLs
exactly as supplied; never translate or rewrite the evidence.
Return only valid JSON:
{{"verdict":"TRUE | FALSE | INSUFFICIENT_EVIDENCE","confidence":0.0,"explanation":"short explanation"}}
TRUE means the evidence directly supports the claim. FALSE means it directly
contradicts the claim. Use INSUFFICIENT_EVIDENCE when it does not decide.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}
"""
    response = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )
    content = response.choices[0].message.content
    result = _parse_json(content)
    legacy = str(result.get("verdict", "")).upper()
    if legacy == "VERIFIED":
        result["verdict"] = "TRUE"
    elif legacy == "MISLEADING":
        result["verdict"] = "INSUFFICIENT_EVIDENCE"
    if result.get("verdict") not in {"TRUE", "FALSE", "INSUFFICIENT_EVIDENCE"}:
        raise ValueError("Hugging Face returned an unsupported verdict")
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    explanation = str(result.get("explanation") or "").strip()
    if not explanation or NON_ENGLISH_MARKERS.search(explanation):
        explanation = {
            "TRUE": "The provided evidence supports the claim.",
            "FALSE": "The provided evidence contradicts the claim.",
            "INSUFFICIENT_EVIDENCE": "The provided evidence is insufficient to determine the claim.",
        }[result["verdict"]]
    result["explanation"] = explanation
    return result


def main() -> int:
    claim = "India became independent in 1989."
    evidence = "India became independent from British rule on 15 August 1947."
    print(json.dumps(verify_claim(claim, evidence), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
