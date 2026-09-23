"""Live web and local retrieval for factual claims."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import requests

from schemas import RetrievalOutput, EvidenceChunk, DetectedClaim, SourceSelectionOutput
from rag.evidence_extractor import rank_passages
from rag.vector_store import query as vector_query, query_temporary_evidence
from rag.web_fetcher import fetch_webpage

LOGGER = logging.getLogger(__name__)


class WebSearchRetriever(ABC):
    @abstractmethod
    def search(
        self,
        query_text: str,
        allowed_domains: list[str],
        n_results: int = 3,
        *,
        request_id: str | None = None,
        claim_id: str | None = None,
        domain: str | None = None,
    ) -> list[dict]:
        """Return dictionaries containing source_name, source_url, text and relevance_score."""
        raise NotImplementedError


class NullWebSearchRetriever(WebSearchRetriever):
    def search(self, query_text, allowed_domains, n_results=3, **kwargs):
        return []


class _SearchResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._anchor: dict | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self._anchor = {"href": attrs.get("href", ""), "title": ""}

    def handle_data(self, data):
        if self._anchor is not None:
            self._anchor["title"] += data.strip() + " "

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor is not None:
            self.results.append(
                {"href": self._anchor["href"], "title": self._anchor["title"].strip()}
            )
            self._anchor = None


def _unwrap_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc:
        target = parse_qs(parsed.query).get("uddg", [])
        if target:
            return unquote(target[0])
    return url


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def source_quality_score(url: str, allowed_domains: list[str] | None = None) -> float:
    domain = _domain(url)
    allowed = [value.lower().removeprefix("www.") for value in (allowed_domains or [])]
    if allowed and any(domain == value or domain.endswith("." + value) for value in allowed):
        return 1.0
    if domain.endswith((".gov", ".gov.uk", ".gov.in", ".edu", ".ac.uk", ".ac.in")):
        return 0.98
    if domain.endswith(".org"):
        return 0.82
    return 0.58


def _select_fetch_results(
    query_text: str,
    results: list[dict],
    allowed_domains: list[str],
    limit: int = 5,
) -> list[dict]:
    stop_words = {"a", "an", "and", "are", "be", "in", "is", "of", "on", "the", "to"}
    query_terms = {
        token for token in re.findall(r"\b[\w'-]{3,}\b", query_text.lower())
        if token not in stop_words
    }
    scored = []
    for index, result in enumerate(results):
        url = result.get("href", "")
        title_terms = set(re.findall(r"\b[\w'-]{3,}\b", result.get("title", "").lower()))
        overlap = len(query_terms & title_terms) / max(1, len(query_terms))
        quality = source_quality_score(url, allowed_domains)
        scored.append((0.7 * overlap + 0.3 * quality, index, result))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[dict] = []
    seen_domains: set[str] = set()
    for _, _, result in scored:
        domain = _domain(result.get("href", ""))
        if domain and domain not in seen_domains:
            selected.append(result)
            seen_domains.add(domain)
        if len(selected) >= limit:
            return selected
    for _, _, result in scored:
        if result not in selected:
            selected.append(result)
        if len(selected) >= limit:
            break
    return selected


def _fetch_pages(results: list[dict], timeout: int) -> list[tuple[dict, str]]:
    def fetch(result: dict) -> tuple[dict, str]:
        url = result.get("href", "")
        LOGGER.info("[FETCH] Fetching: %s", url)
        try:
            page_text = fetch_webpage(url, timeout=timeout)
            if not page_text:
                LOGGER.info("[FETCH] Skipped: %s", url)
                return result, ""
            LOGGER.info("[FETCH] Successfully extracted %d characters", len(page_text))
            return result, page_text
        except Exception:
            LOGGER.warning("[FETCH] Result failed; continuing: %s", url, exc_info=True)
            return result, ""

    with ThreadPoolExecutor(max_workers=min(5, max(1, len(results)))) as executor:
        return list(executor.map(fetch, results))


class LiveWebSearchRetriever(WebSearchRetriever):
    """DuckDuckGo-backed retrieval with actual page fetching.

    No URL catalogue is maintained: every candidate URL comes from a search
    performed for the current claim, then is fetched and passage-ranked.
    """

    def __init__(self, endpoint: str | None = None, timeout: int = 12):
        self.endpoint = endpoint or "https://html.duckduckgo.com/html/"
        self.timeout = timeout

    def _search_urls(self, query_text: str, allowed_domains: list[str], n_results: int) -> list[dict]:
        query = f"fact check {query_text.strip()}"
        if allowed_domains:
            query += " " + " ".join(f"site:{domain}" for domain in allowed_domains)
        try:
            from ddgs import DDGS

            results = DDGS().text(query, max_results=max(n_results * 3, 8))
            output = []
            seen = set()
            for result in results:
                href = _unwrap_url(result.get("href") or result.get("url") or "")
                if href and href not in seen:
                    seen.add(href)
                    output.append({"href": href, "title": result.get("title", "")})
            return output
        except Exception as exc:
            LOGGER.warning("DDGS search unavailable; trying HTML search query=%r error=%s", query, exc)
        try:
            response = requests.get(
                self.endpoint,
                params={"q": query, "kl": "wt-wt"},
                timeout=self.timeout,
                headers={"User-Agent": "TruthBridge/0.1"},
            )
            response.raise_for_status()
            parser = _SearchResultParser()
            parser.feed(response.text)
            output = []
            seen = set()
            for result in parser.results:
                url = _unwrap_url(result["href"])
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                output.append(result | {"href": url})
                if len(output) >= max(n_results * 3, 8):
                    break
            return output
        except requests.RequestException as exc:
            LOGGER.warning("Web search failed query=%r error=%s", query_text, exc)
            return []
        except Exception as exc:
            LOGGER.warning("Web search parsing failed query=%r error=%s", query_text, exc)
            return []

    def search(
        self,
        query_text: str,
        allowed_domains: list[str] | None,
        n_results: int = 3,
        *,
        request_id: str | None = None,
        claim_id: str | None = None,
        domain: str | None = None,
    ) -> list[dict]:
        allowed_domains = allowed_domains or []
        LOGGER.info("[SEARCH] Searching web for: %s", query_text)
        fetched_docs: list[dict] = []
        search_results = self._search_urls(query_text, allowed_domains, n_results)
        LOGGER.info("[SEARCH] Found %d results", len(search_results))
        selected_results = _select_fetch_results(query_text, search_results, allowed_domains)
        LOGGER.info("[FETCH] Selected %d diverse URLs", len(selected_results))
        for result, page_text in _fetch_pages(selected_results, self.timeout):
            url = result.get("href", "")
            if page_text:
                passages = rank_passages(query_text, page_text, max_passages=2)
                if not passages:
                    continue
                LOGGER.info("[CHUNK] Created %d chunks from %s", len(passages), url)
                quality = source_quality_score(url, allowed_domains)
                for index, (passage, relevance) in enumerate(passages):
                    fetched_docs.append(
                        {
                            "id": f"{url}#{index}",
                            "chunk_id": f"{claim_id or 'claim'}_{len(fetched_docs)}",
                            "claim_id": claim_id,
                            "domain": domain or "other",
                            "source_name": result.get("title") or _domain(url),
                            "source_url": url,
                            "text": passage,
                            "relevance_score": 0.82 * relevance + 0.18 * quality,
                        }
                    )
        if not fetched_docs and allowed_domains:
            LOGGER.info(
                "[SEARCH] Restricted sources produced no fetchable evidence; retrying an unrestricted live search"
            )
            fallback_results = self._search_urls(query_text, [], n_results)
            LOGGER.info("[SEARCH] Fallback found %d results", len(fallback_results))
            seen_urls = {item.get("source_url") for item in fetched_docs}
            fallback_results = [
                result for result in _select_fetch_results(query_text, fallback_results, [], limit=5)
                if result.get("href") not in seen_urls
            ]
            for result, page_text in _fetch_pages(fallback_results, self.timeout):
                url = result.get("href", "")
                if not page_text:
                    continue
                passages = rank_passages(query_text, page_text, max_passages=2)
                LOGGER.info("[CHUNK] Created %d chunks from %s", len(passages), url)
                quality = source_quality_score(url, [])
                for index, (passage, relevance) in enumerate(passages):
                    fetched_docs.append(
                        {
                            "id": f"{url}#{index}",
                            "chunk_id": f"{claim_id or 'claim'}_{len(fetched_docs)}",
                            "claim_id": claim_id,
                            "domain": domain or "other",
                            "source_name": result.get("title") or _domain(url),
                            "source_url": url,
                            "text": passage,
                            "relevance_score": 0.82 * relevance + 0.18 * quality,
                        }
                    )
                if len(fetched_docs) >= n_results * 3:
                    break
        ranked = query_temporary_evidence(
            query_text,
            fetched_docs,
            n_results=max(n_results * 2, n_results),
            request_id=request_id,
            claim_id=claim_id,
            domain=domain,
        )
        LOGGER.info("[RAG] Retrieved top %d evidence chunks", len(ranked[:n_results]))
        if ranked:
            LOGGER.info("[RAG] Best source: %s", ranked[0].get("source_url"))
        deduped = []
        seen_fingerprints = set()
        source_counts: dict[str, int] = {}
        for item in ranked:
            fingerprint = " ".join(item.get("text", "").lower().split())
            if fingerprint in seen_fingerprints:
                continue
            source_url = item.get("source_url", "")
            if source_counts.get(source_url, 0) >= 1:
                continue
            seen_fingerprints.add(fingerprint)
            source_counts[source_url] = source_counts.get(source_url, 0) + 1
            deduped.append(item)
        return deduped[:n_results]


class RetrievalAgent:
    def __init__(self, web_search_retriever: WebSearchRetriever | None = None):
        self.web_search = web_search_retriever or LiveWebSearchRetriever()

    def retrieve(
        self,
        claim: DetectedClaim,
        source_selection: SourceSelectionOutput,
        request_id: str | None = None,
    ) -> RetrievalOutput:
        evidence: list[EvidenceChunk] = []
        for source in source_selection.sources:
            try:
                if source.type == "local_rag":
                    collection = self._collection_for(source.name, source_selection.domain)
                    hits = vector_query(collection, claim.text, n_results=3)
                elif source.type == "web_search":
                    from trusted_sources import get_sources_for_domain

                    cfg = next(
                        (s for s in get_sources_for_domain(source_selection.domain) if s["name"] == source.name),
                        {"allowed_domains": []},
                    )
                    try:
                        hits = self.web_search.search(
                            claim.text,
                            cfg.get("allowed_domains", []),
                            n_results=3,
                            request_id=request_id,
                            claim_id=claim.claim_id,
                            domain=source_selection.domain.value,
                        )
                    except TypeError:
                        # Preserve compatibility with custom retrievers using
                        # the original three-argument interface.
                        hits = self.web_search.search(
                            claim.text, cfg.get("allowed_domains", []), n_results=3
                        )
                else:
                    hits = []
                evidence.extend(EvidenceChunk(**hit) for hit in hits if hit.get("text"))
            except Exception:
                LOGGER.exception("Retriever backend skipped source=%s claim=%s", source.name, claim.claim_id)

        evidence.sort(key=lambda item: item.relevance_score, reverse=True)
        deduped: list[EvidenceChunk] = []
        seen: set[tuple[str, str]] = set()
        for item in evidence:
            key = (item.source_url or "", item.text)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        LOGGER.info("Evidence assembled claim=%s count=%d", claim.claim_id, len(deduped))
        return RetrievalOutput(claim_id=claim.claim_id, evidence=deduped[:6])

    @staticmethod
    def _collection_for(source_name: str, domain) -> str:
        from trusted_sources import get_sources_for_domain

        for source in get_sources_for_domain(domain):
            if source["name"] == source_name and source["type"] == "local_rag":
                return source["collection"]
        return domain.value
