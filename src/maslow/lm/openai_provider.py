"""OpenAI Chat Completions implementation of LMProvider."""
from __future__ import annotations

import os

from openai import OpenAI

from .base import LMProvider


class OpenAIProvider(LMProvider):
    """LM provider backed by OpenAI Chat Completions API.

    Usage:
        provider = OpenAIProvider()                      # uses gpt-4o-mini
        provider = OpenAIProvider(model="gpt-4o")
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-...")

    The API key is read from the OPENAI_API_KEY environment variable by default.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 300,
        stop: list[str] | None = None,
    ) -> str:
        """Wrap a bare prompt as a user message and call the chat API."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 300,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
