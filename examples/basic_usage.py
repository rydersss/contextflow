"""
ContextFlow — Quick Start Example

Demonstrates compressing a long AI coding agent conversation
to fit within a smaller token budget while preserving critical context.
"""

from contextflow.adapters import get_adapter
from contextflow.core.compressor import Message, compress, estimate_tokens


def simulate_claude_code_session() -> str:
    """Simulate a realistic Claude Code conversation with ~50 messages."""
    messages = []

    # Initial task
    messages.append("Human: I need to build a REST API for user authentication with JWT tokens.")
    messages.append("Assistant: I'll help you build that. Let me start by checking the project structure.")
    messages.append("Assistant: I see you have a FastAPI project. I'll create the auth module.")

    # Building the auth module
    messages.append("Assistant: Created `auth/models.py` with User model including id, email, hashed_password, created_at")
    messages.append("Assistant: Created `auth/schemas.py` with LoginRequest, TokenResponse classes")

    # Bug fixing session
    messages.append("Human: Getting 401 on every request, even with valid tokens. Can you debug?")
    messages.append("Assistant: Let me check the token verification logic.")
    messages.append("Assistant: Found the bug! In `auth/utils.py` line 47, the JWT algorithm was hardcoded to 'HS256' but you're using RS256 keys. Fixed.")
    messages.append("Human: Still failing. Now it's a different error — 'Token has expired'.")
    messages.append("Assistant: The token expiry was set to 60 seconds instead of 60 minutes. Changed `timedelta(seconds=60)` to `timedelta(minutes=60)`.")
    messages.append("Human: Works now. Can you add refresh token support?")
    messages.append("Assistant: Sure. I'll add a refresh token endpoint. The flow will be: login → access_token + refresh_token, then POST /auth/refresh to get new tokens.")
    messages.append("Assistant: Added `POST /auth/refresh` endpoint. Refresh tokens stored in DB with rotation for security.")

    # Adding tests
    messages.append("Human: Write tests for the auth flow.")
    messages.append("Assistant: Created `tests/test_auth.py` with 12 test cases covering login, refresh, token expiry, and invalid credentials.")
    messages.append("Assistant: Running tests...")
    messages.append("Assistant: 12/12 tests passing. Coverage: auth module now at 94%.")

    # Refactoring
    messages.append("Human: The auth code is getting messy. Refactor into cleaner structure.")
    messages.append("Assistant: I'll reorganize into: auth/services.py (business logic), auth/dependencies.py (FastAPI deps), auth/routes.py (thin handlers).")
    messages.append("Assistant: Refactored. All tests still pass. Reduced auth module from 4 files to 3, removed 80 lines of duplicate code.")

    # Add more conversation padding to simulate a long session
    for i in range(25):
        messages.append(
            f"Assistant: Continuing work on the API — added endpoint for user profile (step {i}), "
            f"fixed type hints, added input validation, improved error handling."
        )

    return "\n\n".join(messages)


def main():
    print("=" * 60)
    print("ContextFlow — Compression Demo")
    print("=" * 60)

    # Get the adapter
    adapter = get_adapter("claude-code")
    print(f"\n[1] Using adapter: {adapter.name}")

    # Simulate a long conversation
    raw_context = simulate_claude_code_session()
    orig_tokens = estimate_tokens(raw_context)
    print(f"[2] Original context: {orig_tokens:,} tokens, {len(raw_context):,} chars")

    # Extract and score messages
    messages = adapter.extract_messages(raw_context)
    print(f"[3] Extracted {len(messages)} messages")

    # Compress
    result = adapter.compress_context(raw_context, token_budget=2000)
    print(f"[4] Compression complete:")
    print(f"    Original:  {result.original_tokens:,} tokens")
    print(f"    Final:     {result.final_tokens:,} tokens")
    print(f"    Reduction: {result.reduction_pct}%")
    print(f"    Kept:      {result.kept_count} messages")
    print(f"    Pruned:    {result.pruned_count} messages")

    # Show compressed output
    print(f"\n[5] Compressed context (first 500 chars):")
    print("-" * 60)
    print(result.compressed_prompt[:500])
    print("...")
    print("-" * 60)

    # Inject back into agent format
    injected = adapter.inject_context(result)
    print(f"\n[6] Injected format: {estimate_tokens(injected):,} tokens")
    print(f"    Ready for Claude Code input ✓")


if __name__ == "__main__":
    main()
