import asyncio
import re


class TopicGuard:
    def __init__(self, allowed_topics: list[str] | None = None) -> None:
        self.allowed_topics = allowed_topics or [
            "bao ve du lieu ca nhan",
            "nghi dinh 13",
            "tai chinh doanh nghiep",
            "bao cao tai chinh",
            "RAG retrieval evaluation",
        ]
        self.block_patterns = [
            r"ignore (all )?(previous|system) instructions",
            r"jailbreak|\\bDAN\\b|evil ai|no restrictions",
            r"decode this|base64",
            r"hack|malware|phishing",
        ]

    def check(self, text: str) -> tuple[bool, str]:
        low = (text or "").lower()
        for pattern in self.block_patterns:
            if re.search(pattern, low, flags=re.I):
                return False, f"Blocked prompt-injection pattern: {pattern}"
        topic_terms = set()
        for topic in self.allowed_topics:
            topic_terms.update(re.findall(r"[\wÀ-ỹ]+", topic.lower()))
        query_terms = set(re.findall(r"[\wÀ-ỹ]+", low))
        overlap = len(topic_terms & query_terms)
        if overlap > 0 or len(low.strip()) < 8:
            return True, "On topic"
        return False, "I can only answer questions related to the provided RAG corpus and evaluation/guardrail scope."

    async def check_async(self, text: str) -> tuple[bool, str]:
        return await asyncio.to_thread(self.check, text)
