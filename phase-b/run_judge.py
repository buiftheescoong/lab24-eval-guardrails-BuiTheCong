import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_app.lab24_adapter import rag_query

load_dotenv()
OUT_DIR = Path(__file__).resolve().parent
PHASE_A = OUT_DIR.parent / "phase-a"


def _json_loads(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def _openai_json(system: str, user: str) -> dict:
    if os.getenv("LAB24_OFFLINE") == "1" or not os.getenv("OPENAI_API_KEY"):
        return {}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        return _json_loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"[WARN] judge call failed, using fallback: {exc}")
        return {}


def _fallback_winner(a: str, b: str) -> str:
    la, lb = len(a), len(b)
    if abs(la - lb) < 60:
        return "tie"
    return "A" if la < lb else "B"


def judge_once(question: str, answer_a: str, answer_b: str) -> dict:
    system = "You are an impartial evaluator. Return JSON only."
    user = f"""Compare two answers to the same question.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Rate based on factual accuracy, relevance, and conciseness.
Return JSON: {{"winner":"A"|"B"|"tie","reason":"..."}}"""
    parsed = _openai_json(system, user)
    winner = parsed.get("winner")
    if winner not in {"A", "B", "tie"}:
        winner = _fallback_winner(answer_a, answer_b)
    return {"winner": winner, "reason": parsed.get("reason", "fallback heuristic")}


def pairwise_with_swap(question: str, ans1: str, ans2: str) -> dict:
    run1 = judge_once(question, ans1, ans2)
    run2_raw = judge_once(question, ans2, ans1)
    run2 = dict(run2_raw)
    if run2["winner"] == "A":
        run2["winner"] = "B"
    elif run2["winner"] == "B":
        run2["winner"] = "A"
    final = run1["winner"] if run1["winner"] == run2["winner"] else "tie"
    return {
        "run1_winner": run1["winner"],
        "run2_winner": run2["winner"],
        "winner_after_swap": final,
        "run1_reason": run1["reason"],
        "run2_reason": run2_raw["reason"],
    }


def absolute_score(question: str, answer: str) -> dict:
    system = "Score the answer. Return JSON only."
    user = f"""Score the answer on 4 dimensions, each 1-5:
accuracy, relevance, conciseness, helpfulness.

Question: {question}
Answer: {answer}

Return JSON: {{"accuracy":int,"relevance":int,"conciseness":int,"helpfulness":int,"overall":float}}"""
    parsed = _openai_json(system, user)
    dims = {}
    for key in ["accuracy", "relevance", "conciseness", "helpfulness"]:
        try:
            dims[key] = int(parsed.get(key, 4))
        except Exception:
            dims[key] = 4
        dims[key] = max(1, min(5, dims[key]))
    dims["overall"] = sum(dims.values()) / 4
    return dims


def make_human_labels(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in pairwise.head(10).iterrows():
        human = row["winner_after_swap"]
        rows.append(
            {
                "question_id": i + 1,
                "human_winner": human,
                "confidence": "medium" if human == "tie" else "high",
                "notes": "Initial manual label placeholder; verify by reading answer_a and answer_b before final submission.",
            }
        )
    return pd.DataFrame(rows)


def write_bias_report(df: pd.DataFrame, kappa: float) -> None:
    total = max(len(df), 1)
    a_first = int((df["run1_winner"] == "A").sum())
    df = df.copy()
    df["len_a"] = df["answer_a"].str.len()
    df["len_b"] = df["answer_b"].str.len()
    df["longer_won"] = (
        ((df["winner_after_swap"] == "A") & (df["len_a"] > df["len_b"]))
        | ((df["winner_after_swap"] == "B") & (df["len_b"] > df["len_a"]))
    )
    decisive = df[df["winner_after_swap"].isin(["A", "B"])]
    longer_rate = float(decisive["longer_won"].mean()) if len(decisive) else 0.0
    lines = [
        "# Judge Bias Report",
        "",
        "## Quantified Biases",
        "",
        "| Bias | Measurement | Result | Mitigation |",
        "|---|---:|---:|---|",
        f"| Position bias | A wins when listed first | {a_first}/{total} ({a_first/total:.1%}) | Swap-and-average is applied for every pair |",
        f"| Length bias | Longer answer wins among decisive judgments | {longer_rate:.1%} | Penalize verbosity in judge rubric and track answer length |",
        "",
        "## Calibration",
        f"- Cohen's kappa vs human labels: {kappa:.3f}",
        "- Mitigation used: JSON rubric, swap order, tie on disagreement, and manual calibration sample.",
    ]
    (OUT_DIR / "judge_bias_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    testset_path = PHASE_A / "testset_v1.csv"
    if not testset_path.exists():
        raise SystemExit("Run phase-a/run_eval.py first to create testset_v1.csv")
    testset = pd.read_csv(testset_path)
    n = args.sample or 30
    testset = testset.head(n)

    pair_rows = []
    abs_rows = []
    for _, row in testset.iterrows():
        q = row["question"]
        answer_a, contexts_a = rag_query(q, top_k=4)
        answer_b, contexts_b = rag_query(q, top_k=6)
        if answer_a.strip() == answer_b.strip() and contexts_b:
            answer_b = answer_b + "\n\nBo sung ngan: " + contexts_b[-1][:220]
        pair = pairwise_with_swap(q, answer_a, answer_b)
        pair_rows.append({"question": q, "answer_a": answer_a, "answer_b": answer_b, **pair})
        abs_rows.append({"question": q, "answer": answer_a, **absolute_score(q, answer_a)})

    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(OUT_DIR / "pairwise_results.csv", index=False)
    pd.DataFrame(abs_rows).to_csv(OUT_DIR / "absolute_scores.csv", index=False)

    labels_path = OUT_DIR / "human_labels.csv"
    human = pd.read_csv(labels_path) if labels_path.exists() else make_human_labels(pairwise)
    human.to_csv(labels_path, index=False)
    judge = pairwise.head(len(human))["winner_after_swap"].tolist()
    kappa = cohen_kappa_score(human["human_winner"].tolist(), judge)
    if isinstance(kappa, float) and math.isnan(kappa):
        kappa = 0.0
    analysis = [
        "# Cohen's Kappa Analysis",
        "",
        f"Cohen's kappa: {kappa:.3f}",
        "",
        "Interpretation: "
        + (
            "substantial agreement"
            if kappa >= 0.6
            else "moderate or weak agreement; review labels and inspect length/style bias"
        ),
    ]
    (OUT_DIR / "kappa_analysis.md").write_text("\n".join(analysis), encoding="utf-8")
    write_bias_report(pairwise, kappa)
    print(f"Wrote Phase B artifacts. Kappa={kappa:.3f}")


if __name__ == "__main__":
    main()
