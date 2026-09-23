# TruthBridge — Phase 1: Real-Time Fact Verification Pipeline

UNESCO Youth Hackathon 2026 — Information Consumption capability only.
(Information Creation / responsible-communication capability is intentionally out of scope for this phase.)

## 1. Architecture

```
User
  │
  ▼
Browser Extension / Mobile App  (captures raw text: transcript, article, post)
  │  POST /verify  { "text": "...", "source_url": "...", "locale": "en" }
  ▼
┌─────────────────────────── Orchestrator Agent ───────────────────────────┐
│                                                                           │
│  LangGraph StateGraph orchestrator wires the stages below                │
│  1. Claim Detection Agent      → extracts factual claims, drops opinion  │
│  2. Source Selection Agent     → picks trusted sources per claim domain  │
│  3. Retrieval Agent (RAG)      → pulls evidence from those sources       │
│  4. Fact Verification Agent    → judges claim vs evidence                │
│  5. Response Generation Agent  → builds final structured JSON            │
│                                                                           │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
Structured JSON result → rendered in extension/app UI
```

Every agent is a **pure function of structured input → structured output** (Pydantic
models, see `schemas.py`). The Orchestrator is the only component that knows the
pipeline order — agents never call each other directly. This makes each agent:

- independently testable
- independently swappable (e.g. swap the LLM, swap the vector DB, swap a source)
- safely parallelizable (claims are verified concurrently by the LangGraph
StateGraph in `orchestrator.py`)

## 2. Why this shape

- **One responsibility per agent** — each file in `agents/` does exactly one job.
- **JSON contracts, not prose** — agents pass typed Pydantic objects; nothing free-texts
  between agents, so failures are structural (validation errors) not semantic (an LLM
  "forgot" to mention something).
- **Domain → trusted source registry is declarative** (`trusted_sources.py`), so adding
  a new source (e.g. a national statistics office) is a config change, not a code change.
- **RAG is source-scoped, not global** — the Retrieval Agent only searches within the
  sources the Source Selection Agent approved for that claim's domain. This is what
  keeps "trusted sources only" an architectural guarantee rather than a prompt-level hope.

## 3. Project layout

```
truthbridge/
├── main.py                     FastAPI app, single POST /verify endpoint
├── schemas.py                  All inter-agent JSON contracts (Pydantic)
├── trusted_sources.py          Declarative registry: domain -> [sources]
├── agents/
│   ├── orchestrator.py         Pipeline coordinator
│   ├── claim_detection.py      Agent 1
│   ├── source_selection.py     Agent 2
│   ├── retrieval.py            Agent 3 (RAG)
│   ├── fact_verification.py    Agent 4
│   └── response_generation.py  Agent 5
├── rag/
│   ├── retrieval.py             Live search, page fetching, and evidence ranking
│   ├── web_fetcher.py           Safe readable-page extraction
│   ├── embeddings.py            Local all-MiniLM-L6-v2 embeddings
│   ├── vector_store.py          Chroma collections (ephemeral for web evidence)
│   └── ingest.py                Script to load optional local documents into Chroma
├── data/                       Sample seed documents for RAG ingestion
└── requirements.txt
```

## 4. Setup

```bash
cd truthbridge
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python rag/ingest.py        # optional: builds local seed collections
uvicorn main:app --reload
```

The Retrieval Agent performs a fresh `ddgs` web search for each claim, fetches result pages,
extracts passages, and ranks them using local `all-MiniLM-L6-v2` (384-dimensional)
embeddings plus lexical, entity, and source-quality signals. Each request/claim writes
chunks to its own temporary ChromaDB collection with trace metadata
(`claim_id`, `source_url`, `source_name`, `retrieved_at`, `chunk_id`, `domain`) and deletes
that collection after ranking. HTTP errors, redirects, bot challenges, boilerplate, and
timeouts are skipped so later results can still provide evidence. Set
`TRUTHBRIDGE_USE_CHROMA=0` only for a minimal offline installation; Chroma is enabled by
default and falls back safely if its native backend is unavailable.
On Python 3.14/Windows, incompatible torch/Chroma native wheels are guarded
to prevent interpreter crashes; install supported wheels (or use Python 3.12)
to activate the local model and native Chroma path.

LLM reasoning uses **Gemini** (via `google-genai`) when `GEMINI_API_KEY` is explicitly
provided (Claim
Detection, Fact Verification, Response Generation). All Gemini calls go through the
single `LLMClient.generate_json()` method in `llm_client.py` — no other file talks to
the API directly, so switching providers or models again later is a one-file change.
Default model is `gemini-2.5-flash`; override with `TRUTHBRIDGE_MODEL` in `.env` (e.g.
`gemini-3.5-flash` or `gemini-3.6-flash` if your key has access).
Fact Verification uses the reusable Hugging Face adapter in
`agents/test_huggingface.py` when `HF_TOKEN` is configured, with Gemini as a provider
fallback. Tokens are read only from environment variables and are never logged.

Test:
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "The WHO says vaccines cause autism. Also, I think blue is a calming color. Global temperatures have risen by more than 1.1°C since pre-industrial times."}'
```

Run the live RAG smoke test (fresh search, fetch, Chroma ranking, and
provider-backed verification):
```bash
python scripts/verify_three_claims.py
```
It includes the expected FALSE claim “India became independent in 1989”, the
expected TRUE claim dated 15 August 1947, and a current 2026 claim. If no
provider token is configured, verification honestly reports insufficient
evidence rather than fabricating a verdict.

## 5. Extending

- **Add a trusted source**: add one entry to `trusted_sources.py` — no other file changes.
- **Add a claim domain**: add a key to `DOMAIN_SOURCE_MAP` in `trusted_sources.py` and,
  if it needs its own evidence corpus, a matching Chroma collection name.
- **Swap the LLM**: every agent takes a `llm_client` in its constructor — point it at any
  Gemini model string via `TRUTHBRIDGE_MODEL`, no pipeline changes. Swapping providers
  entirely means editing only `llm_client.py`.
- **Live web retrieval**: `rag/retrieval.py` generates a fresh fact-check query and uses the
  `ddgs` provider to discover URLs from the live web; no submitted `source_url` or hardcoded
  URL catalogue is required. A different provider can implement the existing
  `WebSearchRetriever` interface.
- **Phase 2 (Responsible Communication)**: plug in as a sibling pipeline that consumes
  the same `schemas.py` contracts — not built here per your scope, but the JSON-first
  design means it can reuse Claim Detection and Fact Verification directly.

## 6. Verdict taxonomy

`TRUE | FALSE | INSUFFICIENT_EVIDENCE` — enforced at the schema level
(`schemas.py::Verdict`). `VERIFIED` remains accepted as a legacy input alias and is
serialized as `TRUE`.
