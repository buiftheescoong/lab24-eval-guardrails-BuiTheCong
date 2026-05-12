"""
Module 5: Enrichment Pipeline
==============================
Lam giau chunks TRUOC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys
import json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk da duoc lam giau."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# --- Technique 1: Chunk Summarization ---


def summarize_chunk(text: str) -> str:
    """
    Tao summary ngan cho chunk.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 cau).
    """
    try:
        from src.llm_helper import chat
        result = chat(
            "Tom tat doan van sau trong 2-3 cau ngan gon bang tieng Viet.",
            text,
            max_tokens=150,
        )
        if result:
            return result
    except Exception:
        pass

    # Extractive fallback: lay 2 cau dau
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    summary = ". ".join(sentences[:2])
    return summary + "." if summary and not summary.endswith(".") else summary


# --- Technique 2: Hypothesis Question-Answer (HyQA) ---


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate cau hoi ma chunk co the tra loi.

    Args:
        text: Raw chunk text.
        n_questions: So cau hoi can generate.

    Returns:
        List of question strings.
    """
    try:
        from src.llm_helper import chat
        result = chat(
            f"Dua tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. Tra ve moi cau hoi tren 1 dong.",
            text,
            max_tokens=200,
        )
        if result:
            questions = result.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
    except Exception:
        pass

    # Extractive fallback
    words = [w for w in text.split() if len(w) > 3][:5]
    questions = []
    if words:
        questions.append(f"Thong tin ve {' '.join(words[:2])} la gi?")
    if len(words) > 2:
        questions.append(f"Quy dinh lien quan den {words[2]} nhu the nao?")
    if len(words) > 4:
        questions.append(f"Chinh sach ve {words[4]} duoc quy dinh ra sao?")
    return questions[:n_questions]


# --- Technique 3: Contextual Prepend (Anthropic style) ---


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giai thich chunk nam o dau trong document.

    Args:
        text: Raw chunk text.
        document_title: Ten document goc.

    Returns:
        Text voi context prepended.
    """
    try:
        from src.llm_helper import chat
        result = chat(
            "Viet 1 cau ngan mo ta doan van nay nam o dau trong tai lieu va noi ve chu de gi. Chi tra ve 1 cau.",
            f"Tai lieu: {document_title}\n\nDoan van:\n{text}",
            max_tokens=80,
        )
        if result:
            return f"{result}\n\n{text}"
    except Exception:
        pass

    # Fallback
    if document_title:
        return f"Trich tu tai lieu: {document_title}.\n\n{text}"
    return text


# --- Technique 4: Auto Metadata Extraction ---


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tu dong: topic, entities, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    try:
        from src.llm_helper import chat
        result = chat(
            'Trich xuat metadata tu doan van. Tra ve JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
            text,
            max_tokens=150,
            json_mode=True,
        )
        if result:
            # Strip markdown code fences if present
            cleaned = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: basic extraction
    category = "policy"
    if any(w in text.lower() for w in ["mat khau", "vpn", "thiet bi", "it", "bao mat"]):
        category = "it"
    elif any(w in text.lower() for w in ["luong", "phu cap", "chi phi", "thuong", "tai chinh"]):
        category = "finance"
    elif any(w in text.lower() for w in ["nghi phep", "nhan vien", "thu viec", "hr"]):
        category = "hr"
    return {"topic": "general", "entities": [], "category": category, "language": "vi"}


# --- Full Enrichment Pipeline ---


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chay enrichment pipeline tren danh sach chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for chunk in chunks:
        original = chunk["text"]
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")

        summary = ""
        if "summary" in methods or "full" in methods:
            summary = summarize_chunk(original)

        questions: list[str] = []
        if "hyqa" in methods or "full" in methods:
            questions = generate_hypothesis_questions(original)

        enriched_text = original
        if "contextual" in methods or "full" in methods:
            enriched_text = contextual_prepend(original, source)

        auto_meta: dict = {}
        if "metadata" in methods or "full" in methods:
            auto_meta = extract_metadata(original)

        enriched.append(EnrichedChunk(
            original_text=original,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Summary: {summarize_chunk(sample)}")
    print(f"HyQA: {generate_hypothesis_questions(sample)}")
    print(f"Contextual: {contextual_prepend(sample, 'So tay nhan vien 2024')}")
    print(f"Metadata: {extract_metadata(sample)}")
