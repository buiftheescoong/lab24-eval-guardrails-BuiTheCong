import asyncio
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


class OutputGuardAPI:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        start = time.perf_counter()
        low = agent_response.lower()
        unsafe_terms = ["self-harm", "weapon", "malware", "hate", "illegal", "attack someone", "phishing", "violent", "credentials", "dangerous"]
        risky = any(term in low for term in unsafe_terms)
        if os.getenv("LAB24_OFFLINE") == "1" or not self.api_key:
            safe = not any(term in low for term in unsafe_terms)
            return safe, "fallback_rule_safe" if safe else "fallback_rule_unsafe", (time.perf_counter() - start) * 1000
        if not risky and os.getenv("LAB24_FORCE_LLAMA_GUARD", "0") != "1":
            return True, "local_precheck_safe_low_risk", (time.perf_counter() - start) * 1000
        payload = {
            "model": "llama-guard-3-8b",
            "messages": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": agent_response},
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.url, json=payload, headers=headers, timeout=30)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]
            result_low = result.lower()
            is_safe = "safe" in result_low and "unsafe" not in result_low
            return is_safe, result, latency_ms
        except Exception as exc:
            safe = not any(term in low for term in unsafe_terms)
            latency_ms = (time.perf_counter() - start) * 1000
            return safe, f"fallback_after_api_error: {exc}", latency_ms

    async def check_async(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        return await asyncio.to_thread(self.check, user_input, agent_response)
