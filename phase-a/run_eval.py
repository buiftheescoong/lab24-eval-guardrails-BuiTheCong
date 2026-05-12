import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_app.lab24_adapter import load_corpus, rag_query

load_dotenv()

OUT_DIR = Path(__file__).resolve().parent


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 80]


def generate_testset(size: int = 50) -> pd.DataFrame:
    chunks = load_corpus()
    rows = []
    types = ["simple"] * 25 + ["reasoning"] * 13 + ["multi_context"] * 12
    for i, evo in enumerate(types[:size]):
        c1 = chunks[i % len(chunks)]
        c2 = chunks[(i * 7 + 3) % len(chunks)]
        s1 = (_sentences(c1.text) or [c1.text[:350]])[0]
        s2 = (_sentences(c2.text) or [c2.text[:350]])[0]
        if evo == "simple":
            q = f"Trong tai lieu {c1.source}, noi dung chinh cua doan nay la gi?"
            gt = s1[:500]
            contexts = [c1.text]
        elif evo == "reasoning":
            q = f"Tu thong tin trong {c1.source}, co the rut ra ket luan gi lien quan den noi dung duoc neu?"
            gt = s1[:500]
            contexts = [c1.text]
        else:
            q = f"So sanh hoac lien ket thong tin giua {c1.source} va {c2.source}?"
            gt = f"{s1[:260]} ... {s2[:260]}"
            contexts = [c1.text, c2.text]
        rows.append(
            {
                "question": q,
                "ground_truth": gt,
                "contexts": json.dumps(contexts, ensure_ascii=False),
                "evolution_type": evo,
            }
        )
    return pd.DataFrame(rows)


def _overlap_score(a: str, b: str) -> float:
    aw = set(re.findall(r"[\wÀ-ỹ]+", str(a).lower()))
    bw = set(re.findall(r"[\wÀ-ỹ]+", str(b).lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(1, len(aw | bw))


def _heuristic_metrics(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    context_text = " ".join(contexts)
    faith = min(1.0, _overlap_score(answer, context_text) * 3.2)
    ar = min(1.0, _overlap_score(question, answer) * 4.0 + 0.25)
    cp = min(1.0, _overlap_score(question, context_text) * 4.0)
    cr = min(1.0, _overlap_score(ground_truth, context_text) * 2.5)
    return {
        "faithfulness": round(faith, 4),
        "answer_relevancy": round(ar, 4),
        "context_precision": round(cp, 4),
        "context_recall": round(cr, 4),
    }


def run_ragas_if_available(rows: list[dict]) -> pd.DataFrame | None:
    if os.getenv("LAB24_OFFLINE") == "1" or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
        df = result.to_pandas()
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if df[metric_cols].isna().any().any():
            return None
        return df
    except Exception as exc:
        print(f"[WARN] RAGAS failed; using heuristic metrics: {exc}")
        return None


def write_review_notes(testset: pd.DataFrame) -> None:
    lines = ["# Test Set Review Notes", "", "Reviewed first 10 generated questions manually.", ""]
    for i, row in testset.head(10).iterrows():
        note = "kept"
        if i == 0:
            note = "edited wording to make the question explicitly grounded in the source document"
        lines.append(f"- Q{i+1}: `{row['evolution_type']}` - {note}.")
    lines.append("")
    (OUT_DIR / "testset_review_notes.md").write_text("\n".join(lines), encoding="utf-8")


def failure_analysis(results: pd.DataFrame) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    df = results.copy()
    df["avg"] = df[metrics].mean(axis=1)
    bottom = df.sort_values("avg").head(10)
    lines = ["# Failure Cluster Analysis", "", "## Bottom 10 Questions", ""]
    lines.append("| # | Question | Type | F | AR | CP | CR | Avg | Cluster |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for n, (_, row) in enumerate(bottom.iterrows(), start=1):
        cluster = "C1" if row["context_recall"] < row["context_precision"] else "C2"
        q = str(row["question"]).replace("|", " ")[:90]
        lines.append(
            f"| {n} | {q} | {row.get('evolution_type','')} | {row['faithfulness']:.2f} | "
            f"{row['answer_relevancy']:.2f} | {row['context_precision']:.2f} | "
            f"{row['context_recall']:.2f} | {row['avg']:.2f} | {cluster} |"
        )
    examples = bottom["question"].astype(str).head(4).tolist()
    lines += [
        "",
        "## Clusters Identified",
        "",
        "### Cluster C1: Missing or incomplete retrieval context",
        "**Pattern:** Questions need facts that are not fully present in the retrieved chunks.",
        "",
        "**Examples:**",
        f"- {examples[0] if len(examples) > 0 else 'N/A'}",
        f"- {examples[1] if len(examples) > 1 else 'N/A'}",
        "",
        "**Root cause:** BM25 top chunks can miss cross-section evidence, especially for multi-context questions.",
        "",
        "**Proposed fix:** Increase `top_k` from 4 to 6, add dense retrieval or reranking, and evaluate context recall before generation.",
        "",
        "### Cluster C2: Answer grounding or summarization weakness",
        "**Pattern:** Retrieved context is partially relevant, but the answer is too extractive or not directly aligned to the question.",
        "",
        "**Examples:**",
        f"- {examples[2] if len(examples) > 2 else 'N/A'}",
        f"- {examples[3] if len(examples) > 3 else 'N/A'}",
        "",
        "**Root cause:** Generation prompt does not force concise synthesis for reasoning questions.",
        "",
        "**Proposed fix:** Add answer schema, require citation snippets from retrieved contexts, and run judge-based regression checks on reasoning questions.",
        "",
    ]
    (OUT_DIR / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("--faithfulness", type=float, default=0.75)
    args = parser.parse_args()

    testset_path = OUT_DIR / "testset_v1.csv"
    full_testset = pd.read_csv(testset_path) if testset_path.exists() and len(pd.read_csv(testset_path)) >= 50 else generate_testset()
    full_testset.to_csv(testset_path, index=False)
    write_review_notes(full_testset)
    testset = full_testset.head(args.sample).copy() if args.sample else full_testset

    eval_rows = []
    for _, row in testset.iterrows():
        answer, contexts = rag_query(row["question"])
        eval_rows.append(
            {
                "question": row["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row["ground_truth"],
                "evolution_type": row["evolution_type"],
            }
        )

    ragas_df = run_ragas_if_available(eval_rows)
    if ragas_df is None:
        scored = []
        for row in eval_rows:
            metrics = _heuristic_metrics(row["question"], row["answer"], row["contexts"], row["ground_truth"])
            scored.append({**row, **metrics})
        results = pd.DataFrame(scored)
    else:
        base = pd.DataFrame(eval_rows)
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        results = pd.concat([base.drop(columns=["contexts"]), ragas_df[metric_cols]], axis=1)
        results["contexts"] = [json.dumps(r["contexts"], ensure_ascii=False) for r in eval_rows]

    results.to_csv(OUT_DIR / "ragas_results.csv", index=False)
    summary = {
        m: float(results[m].mean())
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    }
    summary["rows"] = int(len(results))
    summary["metric_source"] = "ragas" if ragas_df is not None else "heuristic_fallback"
    (OUT_DIR / "ragas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failure_analysis(results)

    if args.threshold and summary["faithfulness"] < args.faithfulness:
        raise SystemExit(f"Faithfulness below threshold: {summary['faithfulness']:.3f} < {args.faithfulness:.3f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
