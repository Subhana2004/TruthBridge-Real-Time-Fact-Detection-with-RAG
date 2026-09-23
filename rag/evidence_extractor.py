"""Extract and rank factual, claim-relevant body passages."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

from rag.embeddings import embed_texts

STOP_WORDS = {
    "a", "an", "and", "are", "be", "been", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with", "will", "says", "said", "according",
}


def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"\b[\w'-]{3,}\b", text.lower()) if word not in STOP_WORDS]


def _entities(text: str) -> set[str]:
    entities = set(re.findall(r"\b(?:[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})\b", text))
    entities.update(re.findall(r"\b(?:19|20)\d{2}\b", text))
    entities.update(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    return {value.lower() for value in entities}


def _candidate_paragraphs(text: str, window: int) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n{2,}", text)
        if len(re.sub(r"\s+", " ", paragraph).strip()) >= 45
    ]
    candidates: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        candidates.append(paragraph[:window])
        if index + 1 < len(paragraphs):
            combined = f"{paragraph} {paragraphs[index + 1]}"
            if len(combined) <= window:
                candidates.append(combined)
    return candidates


def _factuality_score(paragraph: str) -> float:
    """Reward declarative factual content and suppress title-like fragments."""
    lower = paragraph.lower()
    score = 0.0
    if re.search(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?%?\b", paragraph):
        score += 0.18
    if re.search(
        r"\b(?:is|are|was|were|has|have|had|became|occurred|"
        r"requires?|includes?|costs?|offers?|according)\b",
        lower,
    ):
        score += 0.16
    if paragraph.endswith("?") or re.match(r"^(?:is|are|why|how|what|can|does)\b", lower):
        score -= 0.35
    if len(paragraph.split()) < 10:
        score -= 0.2
    return score


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    containment = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.72 or containment >= 0.86 or SequenceMatcher(None, left, right).ratio() >= 0.82


@lru_cache(maxsize=128)
def _claim_embedding(claim: str) -> tuple[float, ...]:
    return tuple(embed_texts([claim])[0])


def _cheap_relevance(claim_tokens: Counter[str], claim_entities: set[str], paragraph: str) -> float:
    paragraph_tokens = set(_tokens(paragraph))
    keyword = sum(claim_tokens[token] for token in paragraph_tokens if token in claim_tokens)
    keyword_score = keyword / max(1, len(claim_tokens))
    entities = _entities(paragraph)
    entity_score = len(claim_entities & entities) / max(1, len(claim_entities))
    claim_numbers = {token for token in claim_entities if any(char.isdigit() for char in token)}
    number_score = len(claim_numbers & entities) / max(1, len(claim_numbers))
    return (0.58 * keyword_score) + (0.27 * entity_score) + (0.15 * number_score) + _factuality_score(paragraph)


def rank_passages(claim: str, text: str, max_passages: int = 3, window: int = 700) -> list[tuple[str, float]]:
    if not text:
        return []
    paragraphs = _candidate_paragraphs(text, window)
    if not paragraphs:
        return []

    claim_tokens = Counter(_tokens(claim))
    claim_entities = _entities(claim)
    cheap_ranked = sorted(
        ((_cheap_relevance(claim_tokens, claim_entities, paragraph), paragraph) for paragraph in paragraphs),
        key=lambda item: item[0],
        reverse=True,
    )
    # Keep a bounded semantic shortlist. The lexical gate preserves candidates
    # with claim terms/entities while the fallback slots retain diverse factual
    # passages when wording differs from the claim.
    shortlist_size = min(len(cheap_ranked), max(3, min(5, max_passages + 1)))
    shortlisted = [paragraph for _, paragraph in cheap_ranked[:shortlist_size]]
    claim_vector = _claim_embedding(claim)
    vectors = embed_texts(shortlisted)
    scored: list[tuple[float, str]] = []
    for paragraph, vector in zip(shortlisted, vectors):
        paragraph_tokens = set(_tokens(paragraph))
        keyword = sum(claim_tokens[token] for token in paragraph_tokens if token in claim_tokens)
        keyword_score = keyword / max(1, len(claim_tokens))
        entities = _entities(paragraph)
        entity_score = len(claim_entities & entities) / max(1, len(claim_entities))
        semantic = max(0.0, sum(a * b for a, b in zip(claim_vector, vector)))
        factuality = _factuality_score(paragraph)
        score = (0.48 * semantic) + (0.25 * keyword_score) + (0.15 * entity_score) + factuality
        if (keyword or entity_score) and semantic >= 0.25:
            scored.append((score, paragraph[:window]))

    scored.sort(key=lambda item: item[0], reverse=True)
    output: list[tuple[str, float]] = []
    seen: set[str] = set()
    for score, paragraph in scored:
        fingerprint = re.sub(r"\W+", " ", paragraph.lower()).strip()
        if fingerprint in seen or any(_near_duplicate(fingerprint, existing) for existing in seen):
            continue
        seen.add(fingerprint)
        output.append((paragraph, round(float(score), 6)))
        if len(output) >= max_passages:
            break
    return output


def extract_relevant_passages(claim: str, text: str, max_passages: int = 3, window: int = 700) -> list[str]:
    return [passage for passage, _ in rank_passages(claim, text, max_passages, window)]
