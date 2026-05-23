"""Pluggable adapter protocol — define your own agent adapter."""

from .base import AgentAdapter
from .__init__ import ADAPTERS, get_adapter

__all__ = ["AgentAdapter", "ADAPTERS", "get_adapter"]
