"""Claude Code adapter — parses Claude Code conversation format."""

import re
from .base import AgentAdapter
from ..core.compressor import Message


class ClaudeCodeAdapter(AgentAdapter):
    """Adapter for Anthropic's Claude Code CLI agent."""

    name = "claude-code"

    def extract_messages(self, raw_context: str) -> list[Message]:
        """Parse Claude Code conversation into messages.

        Claude Code uses: Human: / Assistant: blocks
        Tool use/output are embedded in Assistant blocks as XML.
        """
        messages = []
        pattern = r"(Human|Assistant|System):\s*(.*?)(?=(?:Human|Assistant|System):|$)"
        for match in re.finditer(pattern, raw_context, re.DOTALL):
            role_map = {"Human": "user", "Assistant": "assistant", "System": "tool"}
            role = role_map.get(match.group(1), "tool")
            content = match.group(2).strip()
            if content:
                messages.append(Message(role=role, content=content))
        return messages

    def inject_context(self, compressed) -> str:
        return (
            f"System: [ContextFlow] {compressed.pruned_count} earlier messages "
            f"summarized — {compressed.reduction_pct}% token reduction.\n\n"
            f"{compressed.compressed_prompt}"
        )


class HermesAdapter(AgentAdapter):
    """Adapter for Hermes Agent — conversation-based agent."""

    name = "hermes"

    def extract_messages(self, raw_context: str) -> list[Message]:
        """Parse Hermes Agent conversation into messages.

        Hermes uses role-labeled messages with optional tool call metadata.
        """
        messages = []
        pattern = r"\[(user|assistant|tool)\]\s*(.*?)(?=\[(?:user|assistant|tool)\]|$)"
        for match in re.finditer(pattern, raw_context, re.DOTALL):
            role = match.group(1)
            content = match.group(2).strip()
            if content:
                messages.append(Message(role=role, content=content))
        return messages

    def inject_context(self, compressed) -> str:
        return (
            f"[system] Context compressed: {compressed.pruned_count} messages pruned, "
            f"context reduced by {compressed.reduction_pct}%.\n\n"
            f"{compressed.compressed_prompt}"
        )


class CursorAdapter(AgentAdapter):
    """Adapter for Cursor AI — uses markdown-style conversation blocks."""

    name = "cursor"

    def extract_messages(self, raw_context: str) -> list[Message]:
        """Parse Cursor conversation — User/Assistant pattern with markdown."""
        messages = []
        pattern = r"##\s*(User|Assistant|Tool Output)\s*\n(.*?)(?=##\s*(?:User|Assistant|Tool Output)|$)"
        for match in re.finditer(pattern, raw_context, re.DOTALL):
            role_map = {"User": "user", "Assistant": "assistant", "Tool Output": "tool"}
            role = role_map.get(match.group(1), "tool")
            content = match.group(2).strip()
            if content:
                messages.append(Message(role=role, content=content))
        return messages

    def inject_context(self, compressed) -> str:
        return (
            f"## Context Compressed\n"
            f"> {compressed.pruned_count} messages summarized ({compressed.reduction_pct}% reduction)\n\n"
            f"{compressed.compressed_prompt}"
        )


# Registry
ADAPTERS = {
    "claude-code": ClaudeCodeAdapter,
    "hermes": HermesAdapter,
    "cursor": CursorAdapter,
}


def get_adapter(name: str) -> AgentAdapter:
    """Get adapter by name. Case-insensitive."""
    name = name.lower().strip()
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name]()
