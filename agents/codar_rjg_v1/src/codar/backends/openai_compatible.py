from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from ..types import LLMResponse
from ..utils import parse_first_json_object
from .base import BaseBackend


class OpenAICompatibleBackend(BaseBackend):
    def __init__(self, provider: str, config: Dict[str, Any]):
        self.name = provider
        self.model = str(config.get("model", ""))
        self.api_key = str(config.get("api_key", ""))
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.timeout = int(config.get("timeout_sec", 120))
        self.temperature = float(config.get("temperature", 0.0))

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _mk_user_content(prompt_text: str, media_items: Optional[List[Dict[str, Any]]]) -> Any:
        if not media_items:
            return prompt_text
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for item in media_items:
            url = item.get("url")
            if not url:
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def complete_json(
        self,
        prompt_text: str,
        prompt_id: str,
        media_items: Optional[List[Dict[str, Any]]] = None,
        temperature_override: Optional[float] = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        temperature = self.temperature if temperature_override is None else float(temperature_override)
        payload_base = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": f"You are CoDAR stage {prompt_id}. Return JSON only."},
                {"role": "user", "content": self._mk_user_content(prompt_text, media_items)},
            ],
        }
        payload = dict(payload_base)
        payload["response_format"] = {"type": "json_object"}
        resp = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout)
        if resp.status_code >= 400 and resp.status_code in {400, 404, 422}:
            # Fallback for endpoints that do not support response_format json_object.
            resp2 = requests.post(url, headers=self._headers(), data=json.dumps(payload_base), timeout=self.timeout)
            resp2.raise_for_status()
            data = resp2.json()
            text = data["choices"][0]["message"]["content"]
            parsed = parse_first_json_object(text)
            usage = data.get("usage", {})
            return LLMResponse(parsed_json=parsed, raw_text=text, usage=usage)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        parsed = parse_first_json_object(text)
        usage = data.get("usage", {})
        return LLMResponse(parsed_json=parsed, raw_text=text, usage=usage)

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
        }
