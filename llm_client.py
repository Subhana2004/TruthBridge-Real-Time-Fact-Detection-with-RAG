"""
llm_client.py
=============
Thin wrapper around the Gemini API used by every agent that needs
LLM reasoning (Claim Detection, Fact Verification, Response Generation).
Centralizing this makes the model swappable in one place — every agent
only ever calls generate_json(), so nothing outside this file knows or
cares which provider is behind it.
"""

import os
import json
try:
    from google import genai
    from google.genai import types
except ImportError:  # LLM is optional for web-only/offline retrieval
    genai = None
    types = None

# gemini-2.5-flash is fast/cheap and a good default for a hackathon.
# Swap to gemini-3.5-flash / gemini-3.6-flash via TRUTHBRIDGE_MODEL if you
# want a newer model and your API key has access.
DEFAULT_MODEL = os.environ.get("TRUTHBRIDGE_MODEL", "gemini-flash-latest")


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        # Keep API startup usable for the web-retrieval-only mode. The client
        # is created only when a key is explicitly supplied by the environment.
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key and genai else None
        self.model = model

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
        """
        Calls Gemini with a system instruction that demands JSON-only output,
        using response_mime_type="application/json" so the API itself
        constrains the output format (no manual markdown-fence stripping
        needed). Raises ValueError if parsing fails (callers should catch
        this and degrade gracefully, e.g. treat as INSUFFICIENT_EVIDENCE
        rather than crash the pipeline).
        """
        if self.client is None:
            raise ValueError("GEMINI_API_KEY is not configured")
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        raw = (response.text or "").strip()
        if not raw:
            raise ValueError("Gemini returned an empty response (possibly blocked by safety filters)")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {e}\nRaw output: {raw[:500]}")
