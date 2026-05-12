import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_app.lab24_adapter import rag_query, rag_query_async
from phase_c_imports import InputGuard, OutputGuardAPI, TopicGuard

OUT_DIR = Path(__file__).resolve().parent


def refuse_response() -> str:
    return "I can only answer safe, in-scope questions grounded in the provided documents."


async def audit_log(user_input: str, answer: str, timings: dict) -> None:
    await asyncio.sleep(0)


async def guarded_pipeline(user_input: str) -> tuple[str, dict]:
    input_guard = guarded_pipeline.input_guard
    topic_guard = guarded_pipeline.topic_guard
    output_guard = guarded_pipeline.output_guard
    timings = {}

    t0 = time.perf_counter()
    pii_task = asyncio.create_task(input_guard.sanitize_async(user_input))
    topic_task = asyncio.create_task(topic_guard.check_async(user_input))
    sanitized, _, _ = await pii_task
    topic_ok, _ = await topic_task
    timings["L1"] = (time.perf_counter() - t0) * 1000
    if not topic_ok:
        return refuse_response(), timings

    t0 = time.perf_counter()
    answer, _ = await rag_query_async(sanitized)
    timings["L2"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    safe, _, _ = await output_guard.check_async(sanitized, answer)
    timings["L3"] = (time.perf_counter() - t0) * 1000
    if not safe:
        return refuse_response(), timings

    asyncio.create_task(audit_log(user_input, answer, timings))
    return answer, timings


guarded_pipeline.input_guard = InputGuard()
guarded_pipeline.topic_guard = TopicGuard()
guarded_pipeline.output_guard = OutputGuardAPI()


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
    return vals[idx]


async def benchmark(n: int = 100) -> None:
    testset_path = OUT_DIR.parent / "phase-a" / "testset_v1.csv"
    if testset_path.exists():
        queries = pd.read_csv(testset_path)["question"].dropna().astype(str).tolist()
    else:
        queries = ["Nghi dinh 13 quy dinh gi ve bao ve du lieu ca nhan?"]
    while len(queries) < n:
        queries.extend(queries)
    queries = queries[:n]

    rows = []
    baseline_total = []
    for q in queries:
        t0 = time.perf_counter()
        rag_query(q)
        baseline_ms = (time.perf_counter() - t0) * 1000
        baseline_total.append(baseline_ms)
        _, timings = await guarded_pipeline(q)
        rows.append({"question": q, "baseline_ms": baseline_ms, **timings, "total_ms": sum(timings.values())})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "latency_benchmark.csv", index=False)
    summary_lines = ["# Latency Benchmark", ""]
    for layer in ["L1", "L2", "L3", "total_ms"]:
        vals = df[layer].dropna().tolist()
        summary_lines.append(
            f"- {layer}: P50={percentile(vals,50):.1f}ms, P95={percentile(vals,95):.1f}ms, P99={percentile(vals,99):.1f}ms"
        )
    overhead = df["total_ms"].mean() - statistics.mean(baseline_total)
    summary_lines.append(f"- Average overhead vs baseline: {overhead:.1f}ms")
    (OUT_DIR / "latency_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=0)
    args = parser.parse_args()
    count = args.benchmark or args.sample or 100
    asyncio.run(benchmark(count))
