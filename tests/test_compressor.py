"""Tests for the ContextFlow compression engine."""

import pytest
from contextflow.core.compressor import (
    Message,
    compress,
    estimate_tokens,
    score_relevance,
    summarize_messages,
)


class TestTokenEstimation:
    def test_basic_estimate(self):
        text = "Hello world, this is a test message for token estimation."
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 20  # ~14 tokens

    def test_empty_string(self):
        assert estimate_tokens("") == 1

    def test_long_text(self):
        text = "word " * 1000  # 5000 chars → ~1250 tokens
        tokens = estimate_tokens(text)
        assert 1200 <= tokens <= 1300


class TestRelevanceScoring:
    def test_recent_messages_score_higher(self):
        msgs = [
            Message(role="user", content="old message"),
            Message(role="user", content="recent message"),
        ]
        scored = score_relevance(msgs)
        assert scored[-1].relevance_score > scored[0].relevance_score

    def test_error_messages_score_high(self):
        msgs = [
            Message(role="user", content="hello"),
            Message(role="tool", content="Error: connection refused"),
            Message(role="user", content="ok"),
        ]
        scored = score_relevance(msgs)
        assert scored[1].relevance_score > scored[0].relevance_score

    def test_code_blocks_score_high(self):
        msgs = [
            Message(role="user", content="plain text"),
            Message(role="assistant", content="```python\nx = 1\n```"),
        ]
        scored = score_relevance(msgs)
        assert scored[1].relevance_score > scored[0].relevance_score


class TestCompression:
    def test_no_compression_needed(self):
        msgs = [Message(role="user", content="hi", token_count=2)]
        result = compress(msgs)
        assert result.reduction_pct == 0.0
        assert result.kept_count == 1

    def test_keep_last_n(self):
        msgs = [Message(role="user", content=f"msg {i}", token_count=10) for i in range(10)]
        result = compress(msgs, token_budget=500, keep_last_n=3)
        assert result.kept_count >= 3

    def test_summary_generated(self):
        msgs = [Message(role="user", content=f"message number {i}", token_count=10) for i in range(20)]
        result = compress(msgs, token_budget=100, keep_last_n=2)
        # Should prune some and generate summary
        assert result.pruned_count > 0
        assert result.compressed_prompt

    def test_reduction_percentage(self):
        msgs = [Message(role="user", content=f"long message number {i} with some extra padding", token_count=50) for i in range(50)]
        result = compress(msgs, token_budget=500, keep_last_n=5)
        assert result.reduction_pct > 0


class TestSummarization:
    def test_empty_summary(self):
        assert summarize_messages([]) == ""

    def test_summary_includes_count(self):
        msgs = [
            Message(role="user", content="test message", token_count=10),
            Message(role="user", content="another message", token_count=10),
        ]
        summary = summarize_messages(msgs)
        assert "2 messages pruned" in summary
