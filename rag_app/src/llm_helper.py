"""LLM helper — dùng Gemini (ưu tiên) hoặc OpenAI làm fallback."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, OPENAI_API_KEY

_gemini_client = None


def _get_gemini():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def chat(system: str, user: str, max_tokens: int = 200, temperature: float = 0.1, json_mode: bool = False) -> str:
    """Single-turn chat. Returns response string."""
    # Try Gemini first
    client = _get_gemini()
    if client:
        try:
            from google.genai import types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system}\n\n{user}",
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"  [Gemini error: {e}]", flush=True)

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client_oa = OpenAI(api_key=OPENAI_API_KEY)
            kwargs: dict = {"model": "gpt-4o-mini", "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], "max_tokens": max_tokens, "temperature": temperature}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client_oa.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [OpenAI error: {e}]", flush=True)

    return ""
