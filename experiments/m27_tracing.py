"""
Milestone 27 — Span-Level Tracing & Observability

WHAT THIS DEMONSTRATES:
    Every call to pipeline.forward() now emits three spans: classify, retrieve,
    reply. This script runs a handful of representative tickets, collects those
    traces, prints a per-span breakdown, and writes the complete trace log to
    traces.jsonl.

    The output answers questions that aggregate metrics (M23–M26) cannot:
        - Which step is the latency bottleneck?
        - Is the classify step consistently fast while reply varies?
        - Does a bug ticket cost more tokens than a feature ticket?
        - Does out_of_scope short-circuit measurably reduce latency?

HOW TO RUN:
    python experiments/m27_tracing.py

WHAT TRACES.JSONL IS FOR:
    Langfuse and Arize Phoenix both accept JSONL trace files. You can drag
    traces.jsonl into either tool to get a visual waterfall diagram per trace,
    token cost breakdowns, and anomaly detection — without changing any code.

FINDINGS FROM EXECUTION (5 sample tickets, qwen2.5:14b, direct mode):

    Cache hits appear as zero-token near-instant spans:
        [e43876ef] export button bug — classify=29ms tokens=0+0, reply=2ms tokens=0+0
    DSPy has its own disk cache independent of the gateway. When a ticket was seen
    in a previous session it returns immediately without calling Ollama, leaving no
    new entry in lm.history. extract_token_usage() reads a stale entry and returns
    zeros. This is a useful signal: zero-token spans identify cache hits without any
    extra instrumentation, letting you measure cache effectiveness from the trace log.

    The short-circuit is visible in the n counts:
        classify  n=5  (every ticket classified)
        retrieve  n=4  (out_of_scope skipped)
        reply     n=4  (out_of_scope skipped)
    No special filtering needed — the missing spans tell the story automatically.

    dspy.Refine retry multiplier made visible:
        feature ticket classify=27,772ms tokens=534+803
    The 803 completion tokens (vs ~300 for other tickets) and the 27s latency
    indicate Refine retried at least once. Each retry adds a full LLM round-trip.
    High completion tokens in the classify span are the reliable signal of a retry:
    the model generated extra reasoning to satisfy the constraint on a second pass.
    This is the M19 production concern (monitor retry rate as a metric) made
    directly observable through tracing.

    Retrieve latency is embedding latency, not FAISS search:
        avg 2,259ms, stddev ~50ms — consistent across all four tickets
    FAISS search over 33 entries is microseconds. The 2.2s floor is the Ollama
    embedding call for the query vector. In production, the embedding model's
    latency is the retrieval floor regardless of index size.

    Token averages include cache-hit zeros when grouped naively:
        classify avg tokens=427+346 — pulled down by the first ticket's 0+0
    Filter to token_spans (spans where prompt_tokens > 0) before averaging to
    get clean cost numbers. The zeros remain useful in the raw trace for cache
    detection; they should be excluded from cost projections.

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

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.classifier import configure_dspy
from app.services.pipeline import TicketPipeline, initialize_retriever
from app.services.tracer import start_trace, write_trace

# ---------------------------------------------------------------------------
# Tickets chosen to cover all four pipeline paths:
#   - bug / feature / feedback  → full classify → retrieve → reply path
#   - out_of_scope              → short-circuit after classify (no retrieve/reply)
#
# Five tickets are enough to reveal per-step patterns without a long wait.
# ---------------------------------------------------------------------------

SAMPLE_TICKETS = [
    ("bug",         "The export button crashes every time I click it."),
    ("feature",     "It would be great if I could schedule reports every week."),
    ("feedback",    "The onboarding flow is confusing. Took me 20 minutes."),
    ("out_of_scope","How has your day been?"),
    ("bug",         "I get a 500 error whenever I try to generate a report."),
]


def run_traced_pipeline(pipeline: TicketPipeline, label: str, ticket: str) -> dict:
    """Run one ticket through the pipeline with an active trace. Returns the trace dict."""
    trace = start_trace(ticket)
    pipeline(ticket=ticket)
    write_trace(trace)
    return trace.to_dict()


def main() -> None:
    configure_dspy()
    initialize_retriever()
    pipeline = TicketPipeline()

    traces = []
    print("Running traced pipeline on sample tickets...\n")
    for expected_label, ticket in SAMPLE_TICKETS:
        t = run_traced_pipeline(pipeline, expected_label, ticket)
        traces.append(t)
        print(f"[{t['trace_id']}] {ticket[:60]}")
        print(f"  total: {t['total_duration_ms']:.0f} ms")
        for span in t["spans"]:
            print(f"  {span['name']:10s}  {span['duration_ms']:6.0f} ms", end="")
            if "label" in span:
                print(f"  label={span['label']}", end="")
            if "prompt_tokens" in span:
                print(f"  tokens={span['prompt_tokens']}+{span['completion_tokens']}", end="")
            print()
        print()

    # Group all spans by name across all traces. A single trace shows one ticket;
    # averages across tickets show where to invest optimization effort — if reply
    # averages 6x longer than classify, shorten the reply prompt, not the classifier.
    # n counts per span name also reveal the short-circuit: classify n=5, retrieve n=4.
    by_span = {}
    for trace in traces:
        for span in trace["spans"]:
            name = span["name"]
            if name not in by_span:
                by_span[name] = []
            by_span[name].append(span)

    print("\n=== AVERAGE SPAN METRICS ===")
    for name, spans in by_span.items():
        avg_duration = sum(s["duration_ms"] for s in spans) / len(spans)
        print(f"  {name:10s}  avg {avg_duration:.0f} ms  (n={len(spans)})", end="")
        token_spans = [s for s in spans if "prompt_tokens" in s]
        if token_spans:
            avg_prompt = sum(s["prompt_tokens"] for s in token_spans) / len(token_spans)
            avg_completion = sum(s["completion_tokens"] for s in token_spans) / len(token_spans)
            print(f"  avg tokens={avg_prompt:.0f}+{avg_completion:.0f}", end="")
        print()

    print(f"\nTraces written to traces.jsonl")


if __name__ == "__main__":
    main()
