import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@dataclass
class Chunk:
    text: str
    source: str


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _chunk_text(text: str, source: str, size: int = 900, overlap: int = 140) -> list[Chunk]:
    clean = re.sub(r"\s+", " ", text).strip()
    chunks: list[Chunk] = []
    if not clean:
        return chunks
    step = max(size - overlap, 1)
    for start in range(0, len(clean), step):
        part = clean[start : start + size].strip()
        if len(part) >= 80:
            chunks.append(Chunk(part, source))
    return chunks


def load_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in DATA_DIR.rglob("*"):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            chunks.extend(_chunk_text(_read_file(path), path.name))
    if not chunks:
        raise RuntimeError(f"No corpus text found under {DATA_DIR}")
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


class LocalRAG:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.chunks = load_corpus()
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def retrieve(self, question: str, top_k: int | None = None) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(question))
        k = top_k or self.top_k
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i].text for i in order]

    def generate(self, question: str, contexts: list[str]) -> str:
        api_key = "" if os.getenv("LAB24_OFFLINE") == "1" else os.getenv("OPENAI_API_KEY", "")
        context = "\n\n".join(contexts[:4])
        if api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the provided context. If the context is insufficient, "
                                "say that the information was not found. Be concise and use Vietnamese."
                            ),
                        },
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                    ],
                    temperature=0.1,
                    max_tokens=260,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                print(f"[WARN] OpenAI generation failed, using extractive fallback: {exc}")
        return contexts[0][:700] if contexts else "Khong tim thay thong tin."

    def query(self, question: str, top_k: int | None = None) -> tuple[str, list[str]]:
        contexts = self.retrieve(question, top_k=top_k)
        return self.generate(question, contexts), contexts


_PIPELINE: LocalRAG | None = None


def build_pipeline(top_k: int = 4) -> LocalRAG:
    global _PIPELINE
    _PIPELINE = LocalRAG(top_k=top_k)
    return _PIPELINE


def rag_query(question: str, top_k: int | None = None) -> tuple[str, list[str]]:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_pipeline()
    return _PIPELINE.query(question, top_k=top_k)


async def rag_query_async(question: str, top_k: int | None = None) -> tuple[str, list[str]]:
    return await asyncio.to_thread(rag_query, question, top_k)
