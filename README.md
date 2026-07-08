# DSPy Ticket Classifier

A learning-focused customer support ticket classifier and reply generator built with
[DSPy](https://dspy.ai/), FastAPI, and FAISS. It demonstrates declarative LLM programming,
RAG-grounded reply generation, LLM-as-a-judge evaluation, CI/CD quality gating, span-level
tracing, and production security patterns (context injection defence, output validation, PII
redaction).

This project is the **application layer** that pairs with [AI Gateway](../Ai%20Gateway/) — in
`LM_MODE=gateway` all LLM calls route through the gateway for auth, logging, rate limiting, and
caching. In `LM_MODE=direct` they go straight to Ollama for experiment and evaluation runs.

---

## What It Does

```
POST /v1/pipeline  (ticket text)
       │
       ├─ Feature retrieval  (FAISS, feature KB)    → grounds the label decision
       ├─ Classify ticket    (dspy.Refine, N=3)      → category + reasoning
       │     └─ out_of_scope short-circuit           → returns canned redirect
       ├─ Resolution retrieval  (FAISS, resolution KB) → finds similar past outcomes
       └─ Generate reply    (dspy.ChainOfThought)    → context-grounded support reply
```

Every step is wrapped in a span (M27) so per-step latency and token cost are recorded.
Retrieved context is moderated for indirect prompt injection (M29) and PII-redacted (M32)
before it ever reaches the LLM. The final reply is validated for unsafe content (M30).

---

## Tech Stack

| Layer        | Choice                                                   |
|--------------|----------------------------------------------------------|
| Language     | Python 3.11+                                             |
| API server   | FastAPI + Uvicorn                                        |
| LLM framework| DSPy (`dspy-ai >= 3.2.1`)                               |
| LLM backend  | Ollama (local) or AI Gateway (production)                |
| Retrieval    | FAISS + `nomic-embed-text` (via Ollama)                  |
| Testing      | pytest + pytest-asyncio + httpx                          |

---

## Project Structure

```
dspy-ticket-classifier/
├── app/
│   ├── main.py                  # FastAPI app — configure_dspy + initialize_retriever at startup
│   ├── config.py                # Pydantic Settings — lm_mode, models, gateway/ollama URLs
│   ├── router/
│   │   ├── classify.py          # POST /v1/classify — single-step classification
│   │   ├── pipeline.py          # POST /v1/pipeline — full RAG pipeline
│   │   └── health.py            # GET /health
│   ├── schemas/
│   │   ├── ticket.py            # ClassifyRequest/Response, VALID_LABELS
│   │   └── pipeline.py          # PipelineRequest/Response
│   └── services/
│       ├── classifier.py        # configure_dspy(), ClassifyTicket Signature, TicketClassifier
│       ├── pipeline.py          # TicketPipeline — composes classify + retrieve + reply
│       ├── retriever.py         # FAISS index over knowledge_base.json + feature_kb.json
│       ├── tracer.py            # Span/Trace dataclasses, ContextVar propagation, JSONL output
│       ├── content_guard.py     # Indirect injection detection on retrieved KB context (M29)
│       ├── output_guard.py      # Content validators on LLM output (M30)
│       └── pii_redactor.py      # PII redaction on retrieved context and replies (M32)
├── data/
│   ├── knowledge_base.json      # 33 past resolved tickets — used by the resolution retriever
│   ├── feature_kb.json          # Product feature descriptions — used by the feature retriever
│   └── golden_dataset.json      # 34 labeled examples — used by quality gates
├── experiments/
│   ├── m17_baseline.py          # Jaccard baseline on synthetic dataset
│   ├── m18_optimize.py          # BootstrapFewShot / MIPROv2 — saves optimized_classifier.json
│   ├── m19_constraints.py       # dspy.Refine constraint demo
│   ├── m20_pipeline.py          # Multi-module pipeline without RAG
│   ├── m21_retriever.py         # FAISS retriever smoke test
│   ├── m22_rag_pipeline.py      # RAG pipeline end-to-end demo
│   ├── m23_golden_eval.py       # Three evaluators: exact category, keyword, exact reply
│   ├── m24_llm_judge.py         # LLM-as-judge: score + reason via Instructor
│   ├── m25_rag_triad.py         # Context Relevance, Faithfulness, Answer Relevance
│   ├── m27_tracing.py           # Span-level trace demo
│   └── m28_feature_kb.py        # Two-retriever architecture demo
├── tests/
│   ├── test_quality_gates.py    # M26 — pytest quality gates, session-scoped fixture
│   ├── test_content_guard.py    # Unit tests for indirect injection detection
│   ├── test_output_validation.py# Unit tests for output content validators
│   └── test_pii_redactor.py     # Unit tests for PII redaction
├── traces.jsonl                 # Span-level trace log (appended at runtime)
├── requirements.txt
└── requirements-dev.txt
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally with:
  - A chat model pulled: `ollama pull llama3.1`
  - The embedding model pulled: `ollama pull nomic-embed-text`

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test deps
```

### 4. Configure environment

Create a `.env` file in the project root:

```env
# "direct" → Ollama straight; "gateway" → route through AI Gateway
LM_MODE=direct

OLLAMA_API_BASE=http://localhost:11434
CLASSIFIER_MODEL=ollama/llama3.1
EMBED_MODEL=nomic-embed-text

# Only needed when LM_MODE=gateway
# GATEWAY_URL=http://localhost:8000/v1
# GATEWAY_API_KEY=gw-your-key
```

### 5. Start the server

```bash
uvicorn app.main:app --reload
```

The FAISS index is built at startup from `data/knowledge_base.json` (embeds 33 resolutions — takes a few seconds). The server starts on `http://localhost:8000`.

---

## API Reference

### `POST /v1/classify`

Single-step ticket classification.

```json
{ "ticket": "The export button does nothing when I click it." }
```

Response:
```json
{
  "labels": ["bug"],
  "reasoning": "The user reports a non-functional UI action, which is a software defect."
}
```

Valid labels: `bug`, `feature`, `feedback`, `out_of_scope`.

### `POST /v1/pipeline`

Full RAG pipeline — classify + retrieve + grounded reply.

```json
{ "ticket": "I keep getting a 500 error on the billing page." }
```

Response:
```json
{
  "labels": ["bug"],
  "reply": "Thank you for reporting this. Based on similar issues we've resolved, ...",
  "retrieved_context": ["Past resolution: ..."],
  "reasoning": "..."
}
```

`out_of_scope` tickets return a canned redirect immediately — no retrieval or reply generation is performed.

### `GET /health`

Returns `{"status": "ok"}`.

---

## LM Modes

| Mode      | `LM_MODE` value | When to use |
|-----------|-----------------|-------------|
| `direct`  | `direct`        | **Experiments and evaluation** — bypasses gateway semantic cache, which collapses evaluation scores to 1/N for similar inputs |
| `gateway` | `gateway`       | Production — gains auth, logging, rate limiting, and cost tracking from the AI Gateway |

> **Important:** Always set `LM_MODE=direct` when running `dspy.Evaluate`, any `m1x_*.py` script, or the quality gates. The gateway cache is intentionally aggressive and will corrupt evaluation metrics.

---

## Quality Gates

The quality gate suite (`tests/test_quality_gates.py`) runs the full pipeline against 34 golden examples and asserts that all metrics stay above their thresholds.

```bash
pytest tests/test_quality_gates.py -v
pytest tests/test_quality_gates.py -v -k test_category_accuracy  # single gate
```

| Gate | Threshold | What a failure means |
|------|-----------|----------------------|
| `test_category_accuracy` | ≥ 88% | Classifier Signature or label taxonomy changed |
| `test_judge_quality` | avg ≥ 3.8, ≥4 rate ≥ 75% | Reply Signature or KB content degraded |
| `test_out_of_scope_rate` | ≥ 70% | out_of_scope boundary is eroding |
| `test_rag_triad` | CR ≥ 4.0, F ≥ 4.0, AR ≥ 3.8 | Retriever, grounding, or reply drift |

The `pipeline_results` fixture has `scope="session"` — the 34-example pipeline run is shared across all four test functions (68 LLM calls instead of 272). Expect the full suite to take ~15 minutes.

> **Never lower a threshold to make a gate pass.** Investigate the root cause (Signature description, KB pollution, golden dataset coverage) before touching any threshold.

---

## Experiments

The `experiments/` folder contains standalone scripts for each learning milestone. Run them individually:

```bash
python experiments/m18_optimize.py     # BootstrapFewShot → saves optimized_classifier.json
python experiments/m22_rag_pipeline.py # RAG pipeline demo
python experiments/m24_llm_judge.py    # LLM-as-judge evaluation
python experiments/m25_rag_triad.py    # Context Relevance / Faithfulness / Answer Relevance
python experiments/m27_tracing.py      # Span-level trace demo
python experiments/m28_feature_kb.py   # Two-retriever architecture demo
```

All experiment scripts require `LM_MODE=direct`.

---

## Running Tests

```bash
pytest                                      # all tests
pytest tests/test_quality_gates.py -v      # quality gates (slow, makes LLM calls)
pytest tests/test_pii_redactor.py -v       # fast unit tests
```

`asyncio_mode = auto` is set in `pytest.ini` — no `@pytest.mark.asyncio` needed.

---

## Security

| Layer | What it protects |
|-------|-----------------|
| **Content guard (M29)** | Inspects retrieved KB context for hidden instruction injection before it reaches any DSPy predictor |
| **Output validators (M30)** | Rejects LLM outputs containing HTML tags, SQL keywords, shell metacharacters, or abnormally long content |
| **PII redactor (M32)** | Strips emails, phone numbers, and account IDs from retrieved context and final replies |

The primary PII control is at the retrieval boundary (before the LLM sees the data), not on the output.

---

## Milestones Implemented

| # | Feature |
|---|---------|
| M16 | DSPy Signatures and Modules — `ClassifyTicket`, `TicketClassifier` |
| M17 | Dataset construction and Jaccard baseline |
| M18 | BootstrapFewShot / MIPROv2 optimization |
| M19 | `dspy.Refine` constraint enforcement (replaces deprecated `dspy.Assert`) |
| M20 | Multi-module pipeline — classify + reply in one `dspy.Module` |
| M21 | RAG foundation — FAISS index over `knowledge_base.json` |
| M22 | RAG pipeline — grounded reply with retrieved past resolutions |
| M23 | Golden dataset + deterministic evaluation |
| M24 | LLM-as-a-judge with position-bias mitigation |
| M25 | RAG Triad — Context Relevance, Faithfulness, Answer Relevance |
| M26 | CI/CD quality gating — pytest thresholds over golden dataset |
| M27 | Span-level tracing — per-step latency and cost, JSONL output |
| M28 | Two-retriever architecture — feature KB grounds label decisions |
| M29 | Indirect prompt injection defence on retrieved context (OWASP LLM01) |
| M30 | Output content validation — HTML/SQL/shell pattern rejection (OWASP LLM05) |
| M32 | PII detection and redaction at the retrieval boundary (OWASP LLM06) |
