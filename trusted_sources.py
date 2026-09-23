"""
trusted_sources.py
===================
Declarative registry: ClaimDomain -> list of trusted sources.

Adding a source = adding one entry here. No other file needs to change.
`type` tells the Retrieval Agent which retriever backend to use:
  - "local_rag"  -> look up in the Chroma collection named after `collection`
  - "web_search" -> query WebSearchRetriever, restricted to `allowed_domains`
  - "api"        -> call a dedicated API client (implement in rag/ as needed)
"""

from schemas import ClaimDomain

DOMAIN_SOURCE_MAP = {
    ClaimDomain.HEALTH: [
        {"name": "WHO", "type": "local_rag", "collection": "health",
         "allowed_domains": ["who.int"], "priority": 1},
        {"name": "UNICEF", "type": "local_rag", "collection": "health",
         "allowed_domains": ["unicef.org"], "priority": 2},
        {"name": "Academic Publications (PubMed)", "type": "web_search",
         "allowed_domains": ["pubmed.ncbi.nlm.nih.gov"], "priority": 3},
    ],
    ClaimDomain.CLIMATE_ENVIRONMENT: [
        {"name": "IPCC", "type": "local_rag", "collection": "climate",
         "allowed_domains": ["ipcc.ch"], "priority": 1},
        {"name": "NASA", "type": "local_rag", "collection": "climate",
         "allowed_domains": ["nasa.gov"], "priority": 2},
        {"name": "Government Open Data", "type": "web_search",
         "allowed_domains": ["data.gov"], "priority": 3},
    ],
    ClaimDomain.SCIENCE_TECH: [
        {"name": "NASA", "type": "local_rag", "collection": "science",
         "allowed_domains": ["nasa.gov"], "priority": 1},
        {"name": "Academic Publications", "type": "web_search",
         "allowed_domains": ["nature.com", "science.org", "arxiv.org"], "priority": 2},
    ],
    ClaimDomain.POLITICS_GOVERNANCE: [
        {"name": "Reuters", "type": "web_search",
         "allowed_domains": ["reuters.com"], "priority": 1},
        {"name": "Associated Press", "type": "web_search",
         "allowed_domains": ["apnews.com"], "priority": 2},
        {"name": "Government Open Data", "type": "web_search",
         "allowed_domains": ["data.gov"], "priority": 3},
    ],
    ClaimDomain.ECONOMY_FINANCE: [
        {"name": "Reuters", "type": "web_search",
         "allowed_domains": ["reuters.com"], "priority": 1},
        {"name": "Government Open Data", "type": "web_search",
         "allowed_domains": ["data.gov"], "priority": 2},
    ],
    ClaimDomain.EDUCATION: [
        {"name": "UNESCO", "type": "local_rag", "collection": "education",
         "allowed_domains": ["unesco.org"], "priority": 1},
        {"name": "UNICEF", "type": "local_rag", "collection": "education",
         "allowed_domains": ["unicef.org"], "priority": 2},
        {"name": "UNESCO Web Search", "type": "web_search",
         "allowed_domains": ["unesco.org", "unicef.org"], "priority": 3},
    ],
    ClaimDomain.HUMAN_RIGHTS_SOCIAL: [
        {"name": "UNESCO", "type": "local_rag", "collection": "human_rights",
         "allowed_domains": ["unesco.org"], "priority": 1},
        {"name": "UNICEF", "type": "local_rag", "collection": "human_rights",
         "allowed_domains": ["unicef.org"], "priority": 2},
        {"name": "Reuters", "type": "web_search",
         "allowed_domains": ["reuters.com"], "priority": 3},
    ],
    ClaimDomain.HISTORY: [
        {"name": "UNESCO", "type": "local_rag", "collection": "history",
         "allowed_domains": ["unesco.org"], "priority": 1},
        {"name": "Academic Publications", "type": "web_search",
         "allowed_domains": [], "priority": 2},
    ],
    ClaimDomain.GENERAL_NEWS_EVENT: [
        {"name": "Reuters", "type": "web_search",
         "allowed_domains": ["reuters.com"], "priority": 1},
        {"name": "Associated Press", "type": "web_search",
         "allowed_domains": ["apnews.com"], "priority": 2},
    ],
    ClaimDomain.OTHER: [
        {"name": "Reuters", "type": "web_search",
         "allowed_domains": ["reuters.com"], "priority": 1},
        {"name": "Associated Press", "type": "web_search",
         "allowed_domains": ["apnews.com"], "priority": 2},
    ],
}


def get_sources_for_domain(domain: ClaimDomain):
    return DOMAIN_SOURCE_MAP.get(domain, DOMAIN_SOURCE_MAP[ClaimDomain.OTHER])
