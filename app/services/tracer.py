"""
M27 — Span-Level Tracing & Observability

WHAT A TRACE IS:
    A trace is the complete record of one request end-to-end. It contains one
    or more spans. Each span is a named, timed unit of work: "classify", "retrieve",
    "reply". The span tree answers: which step ran, in what order, how long did each
    take, and what did it cost?

    This is the same model used by OpenTelemetry, Jaeger, Zipkin, Langfuse, and
    Arize Phoenix. The concepts transfer directly.

WHY ContextVar INSTEAD OF PASSING THE TRACE EXPLICITLY:
    The trace needs to be accessible inside pipeline.forward() without changing
    every function signature. Python's contextvars.ContextVar gives each async
    task (and each asyncio.to_thread thread) its own slot for the same variable.
    Two concurrent requests each get their own trace without any locking or ID
    collision. A global dict keyed by request ID would work, but you'd need to
    clean it up manually; ContextVar resets automatically when the task exits.

HOW TO USE:
    trace = start_trace(ticket)        # set the active trace for this context
    pipeline(ticket=ticket)            # forward() calls record_span() internally
    write_trace(trace)                 # serialize to JSONL

PRODUCTION NOTES (M27):
    Replace write_trace() with the OpenTelemetry SDK:
        pip install opentelemetry-sdk opentelemetry-exporter-otlp
    The Span and Trace dataclasses here already mirror the OTel data model.
    Migrating is mostly replacing the JSONL append with tracer.start_span()
    from opentelemetry.trace — the mental model transfers directly.

    In a multi-container deployment, each container should push spans to a
    Redis Stream (XADD) or directly to an OTel Collector via OTLP over gRPC.
    The local JSONL file cannot be shared across containers and is lost on
    container restart. Redis Streams persist entries until a consumer group
    acknowledges them, surviving restarts without data loss.

    Carry a request_id from the gateway's request_logs table into every span's
    metadata. That shared ID links the infrastructure view (auth, rate limiting,
    cost from Project 1) with the application view (label, retrieval quality,
    judge score from this project) into a single unified trace — which is what
    makes root-cause analysis fast when something goes wrong in production.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

log = logging.getLogger(__name__)

import dspy

TRACE_LOG = Path(__file__).parent.parent.parent / "traces.jsonl"

_current_trace: ContextVar["Trace | None"] = ContextVar("_current_trace", default=None)


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        # Milliseconds rather than seconds: sub-second LLM latencies (800 ms vs
        # 1 200 ms) are meaningful differences that seconds would obscure.
        # Returns None for unclosed spans so callers can detect them without
        # getting a misleading zero.
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ticket: str = ""
    spans: list[Span] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> float:
        # Sum span durations as a proxy for end-to-end latency.
        # This is exact only for a purely sequential pipeline; parallel spans
        # would double-count overlapping time. Note it when interpreting output.
        return sum(s.duration_ms or 0.0 for s in self.spans)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "ticket": self.ticket[:80],
            "total_duration_ms": round(self.total_duration_ms, 1),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms or 0.0, 1),
                    **s.metadata,
                }
                for s in self.spans
            ],
        }


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def start_trace(ticket: str) -> Trace:
    """Create a new Trace and install it as the active trace for this context."""
    trace = Trace(ticket=ticket)
    _current_trace.set(trace)
    return trace


@contextmanager
def record_span(name: str) -> Generator[Span, None, None]:
    """
    Context manager that times a block of work and records it on the active trace.

    Usage:
        with record_span("classify") as span:
            result = self.classify(ticket=ticket)
            span.metadata["label"] = result.categories[0]

    If no trace is active, the span is still yielded — it is just not recorded
    anywhere. This makes record_span() safe to call in tests that don't start
    a trace, with no import-time side effects.
    """
    span = Span(name=name)
    try:
        yield span
    finally:
        span.end_time = time.time()
        trace = _current_trace.get()
        if trace is not None:
            trace.spans.append(span)


# ---------------------------------------------------------------------------
# Token cost extraction
# ---------------------------------------------------------------------------

def extract_token_usage() -> dict:
    # Reads token counts and cost from dspy.settings.lm.history[-1].
    # Per-span cost is what lets you answer "is classify or reply responsible
    # for the cost increase?" — gateway logs only show per-request totals.
    # Returns zeros silently on missing data: tracing must never crash the pipeline.
    lm = dspy.settings.lm
    if not lm.history:
        log.warning("extract_token_usage called with empty LM history — span will show zero tokens")
        return {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    last = lm.history[-1]
    usage = last.get("usage", {})
    cost = last.get("cost", 0.0)
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_trace(trace: Trace) -> None:
    # JSONL (one JSON object per line) rather than a JSON array: append without
    # reading the file, stream with tail -f, and feed directly into Langfuse or
    # Arize Phoenix which both accept this format natively.
    d = trace.to_dict()
    with open(TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")
