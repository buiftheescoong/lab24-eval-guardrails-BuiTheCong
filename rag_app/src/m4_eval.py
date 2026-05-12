"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_ragas_llm():
    """Tra ve LLM config cho RAGAS -- uu tien Gemini."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import GEMINI_API_KEY
        if GEMINI_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
            return llm, emb
    except Exception as e:
        print(f"  [RAGAS Gemini LLM setup error: {e}]")
    return None, None


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        llm, embeddings = _get_ragas_llm()
        eval_kwargs: dict = {"dataset": dataset, "metrics": [faithfulness, answer_relevancy, context_precision, context_recall]}
        if llm:
            eval_kwargs["llm"] = llm
        if embeddings:
            eval_kwargs["embeddings"] = embeddings

        result = evaluate(**eval_kwargs)
        df = result.to_pandas()

        import math

        def _to_float(val) -> float:
            """Convert scalar, list, or None to float. NaN → 0.0."""
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                f = float(val)
                return 0.0 if math.isnan(f) else f
            if isinstance(val, list):
                vals = [float(v) for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
                return sum(vals) / len(vals) if vals else 0.0
            try:
                f = float(val)
                return 0.0 if math.isnan(f) else f
            except Exception:
                return 0.0

        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_to_float(row.get("faithfulness")),
                answer_relevancy=_to_float(row.get("answer_relevancy")),
                context_precision=_to_float(row.get("context_precision")),
                context_recall=_to_float(row.get("context_recall")),
            ))

        return {
            "faithfulness": _to_float(result["faithfulness"]),
            "answer_relevancy": _to_float(result["answer_relevancy"]),
            "context_precision": _to_float(result["context_precision"]),
            "context_recall": _to_float(result["context_recall"]),
            "per_question": per_question,
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation error: {e}")
        # Trả về placeholder scores khi RAGAS không chạy được
        per_question = [
            EvalResult(
                question=q, answer=a, contexts=c, ground_truth=gt,
                faithfulness=0.0, answer_relevancy=0.0,
                context_precision=0.0, context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for result in eval_results:
        avg_score = (
            result.faithfulness
            + result.answer_relevancy
            + result.context_precision
            + result.context_recall
        ) / 4.0
        scored.append((avg_score, result))

    # Sort ascending → bottom_n first
    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    diagnosis_map = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    failures = []
    for avg_score, result in bottom:
        metric_scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metric_scores, key=lambda k: metric_scores[k])
        diagnosis, fix = diagnosis_map[worst_metric]

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": metric_scores[worst_metric],
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
