"""Pluggable adapter protocol — Base class and public API.

Import from here to define custom adapters, or use the convenience
``get_adapter()`` from ``contextflow.adapters``.
"""

from ..core import AgentAdapter

__all__ = ["AgentAdapter"]
