# CLAUDE.md — DSPy Ticket Classifier

A FastAPI service that classifies customer support tickets using DSPy declarative signatures.
This is the application layer; the AI Gateway is the infrastructure it can route through.

## Commands

```bash
# Create and activate a virtual environment (do this once)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Install runtime deps
pip install -r requirements.txt

# Install test deps
pip install -r requirements-dev.txt

# Copy and fill in environment variables
copy .env.example .env

# Run the dev server
uvicorn app.main:app --reload

# Run all tests
pytest

# Run a single test
pytest tests/routers/test_classify.py::test_name -v
```

`asyncio_mode = auto` is set in `pytest.ini` — no `@pytest.mark.asyncio` needed.

## Architecture

### Project structure

```
dspy-ticket-classifier/
├── app/
│   ├── main.py              # FastAPI app, lifespan (calls configure_dspy on startup)
│   ├── config.py            # Pydantic Settings — lm_mode, model, gateway/ollama URLs
│   ├── router/
│   │   ├── classify.py      # POST /v1/classify — validates request, calls service, guards label
│   │   └── health.py        # GET /health
│   ├── schemas/
│   │   └── ticket.py        # ClassifyRequest, ClassifyResponse, VALID_LABELS
│   └── services/
│       └── classifier.py    # DSPy layer: configure_dspy(), Signature, Module, classify_ticket()
└── tests/
    ├── conftest.py          # AsyncClient fixture
    └── routers/
        └── test_classify.py
```

### Request flow

```
POST /v1/classify
  → classify router: validate ClassifyRequest (Pydantic)
  → classify_ticket(ticket) in services/classifier.py
      → DSPy module runs LM call (direct Ollama or through gateway)
      → returns ClassifyResponse(label, reasoning, model)
  → router: guard label ∈ VALID_LABELS
  → 200 ClassifyResponse
```

### LM modes

`LM_MODE` in `.env` controls which backend DSPy talks to:

| Mode | Setting | When to use |
|------|---------|-------------|
| `direct` | `LM_MODE=direct` | Simpler; bypass the gateway; faster for pure DSPy learning |
| `gateway` | `LM_MODE=gateway` | Routes through auth, logging, rate limiting; demonstrates the full stack |

### DSPy concepts in this project

- `configure_dspy()` — called once at startup; sets the global LM for all DSPy calls
- `ClassifyTicket` — a `dspy.Signature` declaring `ticket → label`; `desc=` on the output field is the primary prompt lever
- `TicketClassifier` — a `dspy.Module` wrapping either `dspy.Predict` or `dspy.ChainOfThought`
- `asyncio.to_thread` — DSPy's LM calls are synchronous; this keeps the FastAPI event loop unblocked

### M16 TODOs (all in `app/services/classifier.py`)

1. Uncomment `import dspy`
2. Implement `configure_dspy()`: build `dspy.LM(...)` and call `dspy.configure(lm=lm)`
3. Define `ClassifyTicket(dspy.Signature)` with `desc=` on both fields
4. Define `TicketClassifier(dspy.Module)` with `__init__` and `forward`
5. Implement `classify_ticket()`: run the module via `asyncio.to_thread`, return `ClassifyResponse`
6. Uncomment `configure_dspy()` call in `app/main.py` lifespan
7. Run the server, POST a ticket, read `dspy.inspect_history(n=1)` in the console

The key learning moment is step 7: seeing the prompt DSPy constructed automatically from
your Signature, without you writing a single instruction string.
