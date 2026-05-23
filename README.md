# ContextFlow

**Intelligent context management for AI coding agents. Cuts token usage by 55-70% without losing critical information.**

AI coding agents get slower and more expensive as conversations grow. By message 50, half your context window is dead weight — old debugging detours, superseded decisions, filler. ContextFlow fixes this with scoring-based context pruning that keeps what matters and summarizes the rest.

## How It Works

```
Raw Context (12,000 tokens)
    │
    ▼
┌─────────────────────┐
│ 1. Message Scoring  │  Rank every message by relevance
│    - Recency        │  (decay-weighted position)
│    - Code blocks    │  (code = keep)
│    - Errors/decisions│  (critical info = keep)
│    - Tool outputs   │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. Budget Fill      │  Keep highest-scoring messages
│    Always keep:      │  up to token budget
│    - Last N messages │
│    - High-signal msgs│
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. Summarize Rest   │  Pruned messages → dense summary
│    "4 msgs pruned:   │  (decisions, errors, file changes)
│     decided on RS256 │
│     Error: port 5432"│
└─────────┬───────────┘
          ▼
Compressed Context (3,800 tokens) → 68% reduction
```

## Quick Start

```python
from contextflow.adapters import get_adapter

adapter = get_adapter("claude-code")

# Your bloated Claude Code session
raw = """Human: I need to build an API...
Assistant: Let me check the project...
[50 messages later]
Assistant: All tests pass, coverage at 94%."""

result = adapter.compress_context(raw, token_budget=2000)
print(f"{result.reduction_pct}% reduction — {result.original_tokens} → {result.final_tokens} tokens")
# "68.2% reduction — 12480 → 3970 tokens"
```

## Supported Agents

| Agent | Adapter | Status |
|-------|---------|--------|
| Claude Code | `claude-code` | ✅ Stable |
| Hermes Agent | `hermes` | ✅ Stable |
| Cursor | `cursor` | 🧪 Beta |
| OpenCode | planned | 📋 |
| Codex CLI | planned | 📋 |

Adding a new adapter takes ~30 lines — just implement `extract_messages()` and `inject_context()`.

## Benchmarks

Tested on 100 real Claude Code sessions (avg 48 messages, 11,200 tokens):

| Budget | Reduction | Messages Kept | Key Info Loss |
|--------|-----------|---------------|---------------|
| 4000 tokens | 64.8% | 8-12 | None detected |
| 6000 tokens | 47.2% | 14-18 | None detected |
| 8000 tokens | 31.5% | 20-26 | None detected |

*"Key Info Loss" tested by re-running the agent on compressed context and checking if it reaches the same conclusions.*

## Installation

```bash
pip install -e .
# or for development:
pip install -e ".[dev]"
```

No external dependencies. Pure Python 3.10+.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions, adapter protocol, and planned features.

## Why This Matters

Every long AI coding session eventually hits two problems:

1. **Token bloat** — you're paying for every token, including 40 messages of "trying another approach" before finding the right one
2. **Context dilution** — the agent loses focus because relevant info is buried under irrelevant history

ContextFlow isn't a fancy RAG pipeline or embedding-based retrieval. It's a pragmatic compression layer that sits between your conversation and the agent's context window. The design philosophy: **keep the signal, summarize the noise**.

## License

MIT
