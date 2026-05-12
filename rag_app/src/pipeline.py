"""Production RAG Pipeline -- Bai tap NHOM: ghep M1+M2+M3+M4."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# Latency tracking
_latency: dict[str, float] = {}


def _llm_generate(query: str, contexts: list[str]) -> str:
    """LLM generation tu context -- tang faithfulness score."""
    context_str = "\n\n".join(contexts)
    try:
        from src.llm_helper import chat
        result = chat(
            "Tra loi CHI dua tren context duoc cung cap. Neu context khong co thong tin → noi 'Khong tim thay thong tin.' Tra loi ngan gon, ro rang bang tieng Viet.",
            f"Context:\n{context_str}\n\nCau hoi: {query}",
            max_tokens=200,
            temperature=0.1,
        )
        if result:
            return result
    except Exception as e:
        print(f"  LLM generation error: {e}")
    return contexts[0] if contexts else "Khong tim thay thong tin."


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    _latency["chunking_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    print("\n[2/4] Enriching chunks (M5)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual", "metadata"])
    _latency["enrichment_ms"] = (time.perf_counter() - t0) * 1000
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  M5 not implemented -- using raw chunks (fallback)")

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)
    _latency["indexing_ms"] = (time.perf_counter() - t0) * 1000

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    results = search.search(query)
    _latency["search_ms"] = (time.perf_counter() - t0) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    _latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    t0 = time.perf_counter()
    answer = _llm_generate(query, contexts)
    _latency["generate_ms"] = (time.perf_counter() - t0) * 1000

    return answer, contexts


def print_latency_breakdown():
    """In bang latency tung buoc -- Bonus +2."""
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"  {'Step':<20} {'Latency (ms)':>15}")
    print("-" * 38)
    step_names = {
        "chunking_ms": "Chunking (M1)",
        "enrichment_ms": "Enrichment (M5)",
        "indexing_ms": "Indexing (M2)",
        "search_ms": "Search (M2, avg)",
        "rerank_ms": "Rerank (M3, avg)",
        "generate_ms": "LLM Generate (avg)",
    }
    total = 0.0
    for key, label in step_names.items():
        val = _latency.get(key, 0.0)
        total += val
        print(f"  {label:<20} {val:>14.1f}")
    print("-" * 38)
    print(f"  {'TOTAL':<20} {total:>14.1f}")


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'OK' if s >= 0.75 else '  '} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)

    print_latency_breakdown()

    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
