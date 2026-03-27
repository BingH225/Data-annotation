from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import LLMResponse


class BaseBackend(ABC):
    name: str = "base"

    @abstractmethod
    def complete_json(
        self,
        prompt_text: str,
        prompt_id: str,
        media_items: Optional[List[Dict[str, Any]]] = None,
        temperature_override: Optional[float] = None,
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError
