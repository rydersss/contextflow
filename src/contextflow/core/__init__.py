"""Agent adapter interfaces — pluggable integration points.

Each adapter handles:
1. Message extraction (agent-specific format → ContextFlow Message)
2. Token estimation (agent-specific tokenizer)
3. Context injection (compressed prompt → agent input)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from .compressor import Message, CompressionResult, compress


class AgentAdapter(ABC):
    """Base adapter for AI coding agents."""

    name: str = "base"

    @abstractmethod
    def extract_messages(self, raw_context: str) -> list[Message]:
        """Parse agent-specific context into normalized Messages."""

    @abstractmethod
    def inject_context(self, compressed: CompressionResult) -> str:
        """Convert compressed result back to agent-compatible format."""

    def compress_context(
        self, raw_context: str, token_budget: Optional[int] = None
    ) -> CompressionResult:
        """Full compress pipeline: extract → compress → (inject handled separately)."""
        messages = self.extract_messages(raw_context)
        for m in messages:
            # Use agent-specific token estimation
            m.token_count = self.estimate_tokens(m.content)
        return compress(messages, token_budget=token_budget or self.default_budget)

    def estimate_tokens(self, text: str) -> int:
        """Override for agent-specific tokenizer."""
        from .compressor import estimate_tokens

        return estimate_tokens(text)

    @property
    def default_budget(self) -> int:
        return 8000
