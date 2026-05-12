import asyncio
import os
import re
import time


VN_PII = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"\b(?:\+84|0)\d{9,10}\b",
    "phone_us": r"\b(?:\+1[-\s]?)?\d{3}[-\s]\d{3,4}\b|\b(?:\+1[-\s]?)?\d{3}[-\s]\d{3}[-\s]\d{4}\b",
    "tax_code": r"\b\d{10}(?:-\d{3})?\b",
    "email": r"\b[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
    "street_address": r"\b\d{1,5}\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,4}\s+(?:Street|St|Road|Rd|Avenue|Ave|Le Loi|Main Street)\b",
    "self_name": r"\b(?:I'm|I am|Customer)\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+){1,3}\b",
    "vn_name_address": r"\b[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+\s+(?:o|ở)\s+\d{1,5}\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+){0,3}\b",
}


class InputGuard:
    def __init__(self) -> None:
        self.analyzer = None
        self.anonymizer = None
        self.use_presidio = os.getenv("LAB24_USE_PRESIDIO", "0") == "1"
        if not self.use_presidio:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except Exception as exc:
            print(f"[WARN] Presidio unavailable; regex-only PII guard: {exc}")

    def scrub_vn(self, text: str) -> tuple[str, bool]:
        found = False
        out = text
        for name, pattern in VN_PII.items():
            out2 = re.sub(pattern, f"[{name.upper()}]", out)
            found = found or out2 != out
            out = out2
        return out, found

    def scrub_ner(self, text: str) -> tuple[str, bool]:
        if not self.use_presidio or not self.analyzer or not self.anonymizer or not text:
            return text, False
        try:
            results = self.analyzer.analyze(text=text, language="en")
            out = self.anonymizer.anonymize(text=text, analyzer_results=results).text
            return out, bool(results)
        except Exception:
            return text, False

    def sanitize(self, text: str) -> tuple[str, bool, float]:
        start = time.perf_counter()
        out, found_regex = self.scrub_vn(text or "")
        out, found_ner = self.scrub_ner(out)
        latency_ms = (time.perf_counter() - start) * 1000
        return out, found_regex or found_ner, latency_ms

    async def sanitize_async(self, text: str) -> tuple[str, bool, float]:
        return await asyncio.to_thread(self.sanitize, text)
