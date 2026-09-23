"""Offline contract tests for the three representative live-RAG claims."""

from unittest.mock import patch

from rag.retrieval import LiveWebSearchRetriever


CLAIMS = [
    "India became independent from British rule in 1947.",
    "Global surface temperature was about 1.1 degrees Celsius above pre-industrial levels.",
    "Vaccines train the immune system to create antibodies.",
]


def test_three_claims_are_fetched_ranked_and_return_evidence():
    pages = {
        "https://example.org/india": "India became independent from British rule on 15 August 1947. This page contains historical information.",
        "https://example.org/climate": "Global surface temperature was about 1.1 degrees Celsius above the 1850 to 1900 average during 2011 to 2020. This is a climate report.",
        "https://example.org/vaccines": "Vaccines train the immune system to create antibodies and protect people from disease. This is health information.",
    }

    retriever = LiveWebSearchRetriever()
    for claim, url in zip(CLAIMS, pages):
        with patch.object(retriever, "_search_urls", return_value=[{"href": url, "title": "Example source"}]):
            with patch("rag.retrieval.fetch_webpage", return_value=pages[url]):
                evidence = retriever.search(claim, [], n_results=1)
        assert evidence
        assert evidence[0]["source_url"] == url
        assert evidence[0]["text"]
        assert isinstance(evidence[0]["relevance_score"], float)


def test_bad_search_results_are_skipped():
    retriever = LiveWebSearchRetriever()
    with patch.object(
        retriever,
        "_search_urls",
        return_value=[{"href": "javascript:alert(1)", "title": "bad"}],
    ):
        assert retriever.search(CLAIMS[0], [], n_results=3) == []
