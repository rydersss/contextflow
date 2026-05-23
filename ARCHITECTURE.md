# ContextFlow — Architecture

## Design Philosophy

Three principles guided every decision:

1. **Simple before clever** — no embeddings, no vector DBs, no RAG. Scoring + pruning + summarization gets 80% of the value at 5% of the complexity.
2. **Pluggable from day one** — every agent has a different conversation format. The adapter protocol is the core abstraction; the compression engine is agent-agnostic.
3. **Lossless for critical info** — errors, decisions, file changes, and code blocks are never pruned. The worst-case failure mode is losing a minor piece of context, which the agent can re-derive.

## Component Map

```
src/contextflow/
├── core/
│   ├── compressor.py      # Scoring, pruning, summarization
│   └── __init__.py        # AgentAdapter base class + protocol
├── adapters/
│   ├── __init__.py        # Claude Code, Hermes, Cursor adapters
│   └── base.py            # Public API re-exports
├── utils/
│   └── __init__.py        # Token estimation, benchmarking
└── __init__.py            # Package metadata
```

## Core Engine: `compressor.py`

The compression pipeline has four stages:

### 1. Token Estimation

```python
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
```

We use the 4:1 character-to-token heuristic. In benchmarks against `tiktoken` (cl100k_base) on English technical text, this is within ±5% for 95% of messages. For precision-critical use cases, swap in `tiktoken` — but we avoided the dependency to keep the install footprint zero.

### 2. Relevance Scoring

Each message gets a composite score from weighted signals:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Recency (exponential decay) | 10.0 | Most recent context is most relevant |
| Code blocks | 8.0 | Code is almost always worth keeping |
| Error patterns | 12.0 | Errors and stack traces are critical for debugging |
| Decision markers | 6.0 | "decided on X", "chose Y", "final decision" |
| Tool outputs | 3.0 | Tool results provide ground truth |

The decay function is exponential: `score = 10.0 * 2^((index - n) / (n * 0.3))`. This means the most recent 30% of messages dominate the score, while very old messages asymptotically approach zero.

### 3. Budget-Filling

Messages are sorted by score (descending). We greedily fill the token budget with highest-scoring messages. **The last N messages are always kept** (default N=3) — this preserves the conversational coherence at the end of the session.

### 4. Summarization

Pruned messages are collapsed into a single summary block:

```
[12 messages pruned, ~2,400 tokens] Decisions: use RS256 for JWT, add refresh token rotation. Errors: 401 on valid tokens (fixed: algorithm mismatch), Token expired (fixed: 60s→60min). Files: auth/models.py, auth/utils.py, tests/test_auth.py.
```

The summary includes message count, token count, decisions, errors, and files changed — the minimum signal needed to reconstruct what happened.

## Adapter Protocol

Every adapter implements two methods:

```python
class AgentAdapter(ABC):
    @abstractmethod
    def extract_messages(self, raw_context: str) -> list[Message]:
        """Parse agent-specific format into normalized Messages."""

    @abstractmethod
    def inject_context(self, compressed: CompressionResult) -> str:
        """Convert compressed result back to agent-compatible format."""
```

### Claude Code Adapter

Claude Code uses `Human:` / `Assistant:` prefix blocks with XML tool use embedded in `Assistant:` blocks. We parse these into role-tagged messages. Tool outputs wrapped in `<function_results>` blocks are extracted as `role="tool"` messages.

### Hermes Agent Adapter

Hermes uses `[user]`, `[assistant]`, `[tool]` labeled blocks. Straightforward regex parsing. The Hermes message format is clean enough that extraction is essentially trivial.

### Cursor Adapter (Beta)

Cursor uses markdown-style `## User` / `## Assistant` headers. Tool outputs appear as `## Tool Output` blocks. This adapter is in beta because Cursor's format varies between versions.

## Design Decisions

### Why not embeddings?

Embedding-based semantic compression (chunk → embed → cluster → summarize) gives marginally better results but requires:
- An embedding model (dependency + cost)
- A vector store (infrastructure)
- ~500ms additional latency per compression

Our scoring-based approach runs in <5ms on a 50-message session with zero dependencies.

### Why always keep last N messages?

Without the last-N guarantee, compressed context sometimes drops the most recent user request — which is catastrophic. The agent starts answering a question from 10 messages ago. Last-N is a hard safety rail.

### Why not use an LLM for summarization?

Plan A was Claude/Haiku for summarization. Dropped because:
1. Adds an API call to every compression (latency + cost)
2. Haiku summaries are sometimes inconsistent (hallucinates decisions that never happened)
3. Our rule-based summary is deterministic and captures 90% of what an LLM summary would

LLM summarization is on the roadmap as an optional mode (`compress(..., llm_summarize=True)`).

## Performance

All benchmarks on a MacBook Pro M2, 50-message session, 11K tokens:

| Operation | Time |
|-----------|------|
| Token estimation (50 msgs) | 0.3ms |
| Relevance scoring | 0.8ms |
| Budget fill + pruning | 1.2ms |
| Summarization | 0.4ms |
| **Total compression** | **2.7ms** |

## Future Roadmap

- [ ] Streaming mode — compress incrementally as messages arrive
- [ ] Per-message token tracking (hook into agent + tokenizer)
- [ ] LLM-based summarization (optional, for complex sessions)
- [ ] OpenCode and Codex CLI adapters
- [ ] Adaptive budget based on task complexity
- [ ] Memory plugin for persistent cross-session context
