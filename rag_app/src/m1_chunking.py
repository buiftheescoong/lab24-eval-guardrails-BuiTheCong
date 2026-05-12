"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        def cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        chunks: list[Chunk] = []
        current_group = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                chunks.append(Chunk(
                    text=" ".join(current_group),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                ))
                current_group = []
            current_group.append(sentences[i])
        if current_group:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
        return chunks

    except ImportError:
        # Fallback: sentence boundary splitting khi không có sentence-transformers
        chunks = []
        current_group = [sentences[0]]
        for i in range(1, len(sentences)):
            current_group.append(sentences[i])
            if len(" ".join(current_group)) > 400:
                chunks.append(Chunk(
                    text=" ".join(current_group[:-1]),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                ))
                current_group = [sentences[i]]
        if current_group:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
        return chunks if chunks else [Chunk(text=text.strip(), metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_text = ""
    p_index = 0
    for para in paragraphs:
        if len(current_text) + len(para) > parent_size and current_text:
            pid = f"parent_{p_index}"
            parent = Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            )
            parents.append(parent)

            # Split parent into children using sliding window
            ptext = current_text.strip()
            start = 0
            c_index = 0
            while start < len(ptext):
                end = start + child_size
                child_text = ptext[start:end].strip()
                if child_text:
                    children.append(Chunk(
                        text=child_text,
                        metadata={**metadata, "chunk_type": "child", "child_index": c_index},
                        parent_id=pid,
                    ))
                    c_index += 1
                start += child_size // 2  # 50% overlap
                if start >= len(ptext) - 10:
                    break

            p_index += 1
            current_text = ""
        current_text += para + "\n\n"

    # Handle remaining text
    if current_text.strip():
        pid = f"parent_{p_index}"
        parent = Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        )
        parents.append(parent)

        ptext = current_text.strip()
        start = 0
        c_index = 0
        while start < len(ptext):
            end = start + child_size
            child_text = ptext[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "child_index": c_index},
                    parent_id=pid,
                ))
                c_index += 1
            start += child_size // 2
            if start >= len(ptext) - 10:
                break

    # Ensure at least one parent and one child exist
    if not parents and text.strip():
        pid = "parent_0"
        parents.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))
        children.append(Chunk(
            text=text.strip()[:child_size],
            metadata={**metadata, "chunk_type": "child", "child_index": 0},
            parent_id=pid,
        ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Save previous section
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Don't forget last section
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
        ))

    # Fallback nếu không tìm thấy headers
    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "", "strategy": "structure"},
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results: dict[str, dict] = {
        "basic": {"num_chunks": 0, "avg_length": 0, "min_length": float("inf"), "max_length": 0},
        "semantic": {"num_chunks": 0, "avg_length": 0, "min_length": float("inf"), "max_length": 0},
        "hierarchical": {"num_chunks": 0, "avg_length": 0, "min_length": float("inf"), "max_length": 0},
        "structure": {"num_chunks": 0, "avg_length": 0, "min_length": float("inf"), "max_length": 0},
    }
    all_chunks: dict[str, list[Chunk]] = {k: [] for k in results}

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})

        all_chunks["basic"].extend(chunk_basic(text, metadata=meta))
        all_chunks["semantic"].extend(chunk_semantic(text, metadata=meta))
        _, children = chunk_hierarchical(text, metadata=meta)
        all_chunks["hierarchical"].extend(children)
        all_chunks["structure"].extend(chunk_structure_aware(text, metadata=meta))

    print(f"\n{'Strategy':<15} | {'Chunks':>7} | {'Avg Len':>8} | {'Min':>5} | {'Max':>5}")
    print("-" * 50)
    for name, chunks in all_chunks.items():
        if not chunks:
            results[name] = {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
            print(f"{name:<15} | {'0':>7} | {'0':>8} | {'0':>5} | {'0':>5}")
            continue
        lengths = [len(c.text) for c in chunks]
        avg = int(sum(lengths) / len(lengths))
        mn = min(lengths)
        mx = max(lengths)
        results[name] = {"num_chunks": len(chunks), "avg_length": avg, "min_length": mn, "max_length": mx}
        print(f"{name:<15} | {len(chunks):>7} | {avg:>8} | {mn:>5} | {mx:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
