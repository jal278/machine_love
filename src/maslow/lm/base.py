"""Abstract LM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LMProvider(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 300,
        stop: list[str] | None = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 300,
    ) -> str:
        """Generate a chat completion for the given messages."""
        ...
