# CLAUDE.md — DSPy Ticket Classifier

This file provides guidance to Claude Code when working in this repository.

A standalone DSPy application that classifies customer support tickets and generates
grounded replies using RAG. It is the application layer; the AI Gateway (Project 1) is
the infrastructure it routes through in production.

---

## Commands

```bash
# Activate the virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Install deps
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the dev server
uvicorn app.main:app --reload

# Run all tests (includes quality gates — slow, ~15 min, makes LLM calls)
pytest

# Run quality gates only
pytest tests/test_quality_gates.py -v

# Run a single gate
pytest tests/test_quality_gates.py::test_category_accuracy -v

# Run a single experiment
python experiments/m24_llm_judge.py

# Run evaluation baseline
python experiments/m23_golden_eval.py
```

`asyncio_mode = auto` is set in `pytest.ini` — no `@pytest.mark.asyncio` needed.

**Critical for experiment runs:** set `LM_MODE=direct` in `.env`. The gateway's
semantic cache collapses all evaluation scores to 1/N. Direct mode bypasses it.

---

## Architecture

### Module map

```
app/
├── main.py                   # FastAPI app + lifespan (configure_dspy, initialize_retriever)
├── config.py                 # Pydantic Settings — lm_mode, models, gateway/ollama URLs
├── router/
│   ├── classify.py           # POST /v1/classify
│   └── pipeline.py           # POST /v1/pipeline (classify + retrieve + reply)
├── schemas/
│   └── ticket.py             # ClassifyRequest, ClassifyResponse, PipelineResponse
└── services/
    ├── classifier.py         # M16/M18/M19 — configure_dspy(), ClassifyTicket Signature,
    │                         #   TicketClassifier (dspy.Refine wrapper), classify_ticket()
    ├── pipeline.py           # M20/M22/M25.5 — TicketPipeline (classify→retrieve→reply),
    │                         #   out_of_scope short-circuit, initialize_retriever()
    ├── retriever.py          # M21 — FAISS index over knowledge_base.json, cosine similarity
    └── tracer.py             # M27 — Span/Trace dataclasses, ContextVar propagation, JSONL

data/
├── knowledge_base.json       # 33 past resolutions used by the retriever (M21)
└── golden_dataset.json       # 34 labeled examples used by quality gates (M23/M26)

experiments/
├── m17_baseline.py           # Jaccard baseline on synthetic dataset
├── m18_optimize.py           # BootstrapFewShot / MIPROv2 — saves optimized_classifier.json
├── m19_constraints.py        # dspy.Refine constraint demo
├── m20_pipeline.py           # Multi-module pipeline without RAG
├── m21_retriever.py          # FAISS retriever smoke test
├── m22_rag_pipeline.py       # RAG pipeline end-to-end demo
├── m23_golden_eval.py        # Three evaluators: exact category, keyword, exact reply
├── m24_llm_judge.py          # LLM-as-judge: evaluate_response() → ReplyJudgement
├── m25_rag_triad.py          # RAG Triad: evaluate_triad() → CR, F, AR scores
└── m27_tracing.py            # Span-level trace demo

tests/
└── test_quality_gates.py     # M26 — pytest assertions on category accuracy, judge
                              #   quality, out_of_scope detection rate, RAG Triad averages

insights.md                   # Non-obvious learnings from M16–M27 — read before designing
milestones.md                 # Canonical milestone status — update when milestones complete
```

### Pipeline flow (M22+)

```
POST /v1/pipeline
  → TicketPipeline.forward(ticket)
      1. classify       → ClassifyTicket Signature (dspy.Refine, N=3)
      2. short-circuit  → if out_of_scope: return canned redirect, retrieved_context=[]
      3. retrieve       → Retriever.retrieve(ticket, k=3) — FAISS cosine similarity
      4. reply          → RetrievedReply Signature grounded in retrieved context
  → PipelineResponse(labels, reply, retrieved_context, reasoning)
```

Each step is wrapped in `record_span()` from tracer.py so latency and token cost
are tracked per-step when a trace is active.

### LM modes

| Mode | Setting | When to use |
|---|---|---|
| `direct` | `LM_MODE=direct` | Experiment runs and evaluation — bypasses gateway cache |
| `gateway` | `LM_MODE=gateway` | Production — gains auth, logging, rate limiting, cost tracking |

### Key runtime gotchas

- **Gateway semantic cache corrupts evaluations.** Similar inputs return the same cached
  answer, collapsing Jaccard scores to 1/N. Always use `LM_MODE=direct` when running
  `dspy.Evaluate`, `m17_baseline.py`, `m18_optimize.py`, or the quality gates.
- **DSPy LM calls are synchronous.** All LM calls go through `asyncio.to_thread` to avoid
  blocking the FastAPI event loop.
- **`ContextVar` propagates through `asyncio.to_thread`.** The tracer's `_current_trace`
  is automatically available inside `pipeline.forward()` when the trace was started before
  `to_thread` was called — no explicit passing needed.
- **`dspy.Prediction` mirrors Signature output fields.** If a field is named `categories`,
  the prediction has `.categories`. Naming it `labels` would conflict with `dspy.Example.labels()`.

---

## Quality gates

`tests/test_quality_gates.py` wraps all M23–M25 evaluators into pytest assertions.
The suite makes LLM calls for every golden example — budget ~15 minutes per run.

| Gate | Threshold | What it catches |
|---|---|---|
| `test_category_accuracy` | ≥ 88% (non-oos only) | Classifier Signature or label taxonomy changes |
| `test_judge_quality` | avg ≥ 3.8, ≥4 rate ≥ 75% | Reply Signature or KB content changes |
| `test_out_of_scope_rate` | ≥ 70% | out_of_scope boundary eroding |
| `test_rag_triad` | CR ≥ 4.0, F ≥ 4.0, AR ≥ 3.8 | Retriever changes, grounding drift, reply drift |

**Never lower a threshold to make a failing gate pass.** That is equivalent to deleting the
gate. Investigate the root cause first — most failures trace back to the Signature docstring,
the KB content, or the golden dataset labels.

The `pipeline_results` fixture has `scope="session"` — the pipeline runs once and all four
test functions share the results. Without it: 4 tests × 34 examples × 2 LLM calls = 272
calls per run. With it: 68 calls.

---

## Collaboration style

These patterns apply to all work in this repo. They are non-negotiable and override
any default behavior.

**Learning-first.** Never implement milestone logic in full unless explicitly asked.
Create skeletons with TODO comments that explain the *why* of each piece, then guide
the user to implement them. The goal is understanding, not output.

**TODO cleanup pattern.** After the user implements TODOs, they will ask to "refactor
the TODOs" or "remove the scaffolding." When this happens:
- Remove the skeleton code comments and the `raise NotImplementedError` stubs
- Keep and expand any explanatory "why" content in a regular comment block
- Do NOT rewrite or simplify the user's implementation — only remove the scaffolding

**PRODUCTION NOTES pattern.** Each service file and experiment file has a clearly marked
`PRODUCTION NOTES` block explaining what would change in a real deployment. When adding
new files, include this block. When a production consideration comes up in conversation,
append it to the relevant file's block and to `insights.md`.

**insights.md** is the canonical place for non-obvious learnings — not for what the code
does (the code says that), but for why decisions were made, what failed, what the ceiling
is, and what the production gap looks like. Append to it freely; never overwrite entries.

**milestones.md** (in `c:\study projects\`) is the canonical source of truth for milestone
status. Update it when a milestone completes (🔄 → ✅) and when a new one starts (⬜ → 🔄).
Update "Immediate Next Steps" to reflect what's actually next.

**Explain root causes before fixing.** When a quality gate fails or an experiment produces
unexpected results, diagnose which specific examples are likely failing and why before
touching any code. The diagnosis should name the structural cause (ambiguous label boundary,
wrong Signature description, KB pollution) not just the symptom (accuracy dropped).

---

## Data files

### `data/knowledge_base.json`

33 entries. Each entry has `ticket`, `category`, `resolution`. The Retriever embeds
`resolution` (not ticket) so retrieval matches "what was done" rather than "what was said."

Adding entries changes the FAISS embedding space and can shift nearest neighbours for
existing queries (KB pollution). Re-run `m25_rag_triad.py` after any KB expansion to
check for regressions.

### `data/golden_dataset.json`

34 entries. Each has `ticket`, `expected_category`, `expected_reply_keywords`,
`expected_reply`. The 34 entries break down as:
- 9 bug, 8 feature, 9 feedback (standard tickets)
- 3 wrong-premise feedback (customer assumes a feature is missing — it exists)
- 6 out_of_scope (2 obvious, 2 medium, 2 tricky boundary cases)

**Never modify entries to make a gate pass.** Only add entries when new coverage is needed.

---

## DSPy patterns used in this project

| Pattern | Where | Why |
|---|---|---|
| `dspy.ChainOfThought` over `dspy.Predict` | classify, reply steps | Ambiguous inputs benefit from reasoning before committing to a label or reply |
| `dspy.Refine(N=3)` | `TicketClassifier` | Enforces valid label constraint; retries with generated hint on failure |
| `ContextVar` for trace propagation | `tracer.py` | Avoids passing trace object through every function signature |
| `scope="session"` pytest fixture | `test_quality_gates.py` | Runs 34-example pipeline evaluation once, shared across all gate assertions |
| Separate `EvaluateReplyWithHint` Signature | `m24_llm_judge.py` | DSPy rejects unknown kwargs — retry Signatures need explicit `hint: str = dspy.InputField()` |
| `dspy.ChainOfThought(Signature)(...)` | `m25_rag_triad.py` | Signatures are declarations, not callable — must be wrapped in a Module to call |
