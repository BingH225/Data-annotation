from __future__ import annotations

from typing import Any, Dict

from .base import BaseBackend
from .mock_backend import MockBackend
from .openai_compatible import OpenAICompatibleBackend


def create_backend(backend_cfg: Dict[str, Any]) -> BaseBackend:
    provider = str(backend_cfg.get("provider", "mock")).lower().strip()
    if provider == "mock":
        return MockBackend()
    if provider in {"vllm", "openai"}:
        return OpenAICompatibleBackend(provider=provider, config=backend_cfg)
    raise ValueError(f"Unsupported backend provider: {provider}")

