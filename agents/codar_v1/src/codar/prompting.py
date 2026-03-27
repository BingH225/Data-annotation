from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .types import PromptMeta
from .utils import json_dumps, sha256_text


@dataclass
class RenderedPrompt:
    text: str
    meta: PromptMeta


class PromptStore:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir

    def load_template(self, prompt_id: str) -> str:
        path = self.prompts_dir / f"{prompt_id}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, prompt_id: str, prompt_vars: Dict[str, Any]) -> RenderedPrompt:
        tmpl = self.load_template(prompt_id)
        safe_vars = {k: (json_dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in prompt_vars.items()}
        rendered = tmpl.format(**safe_vars)
        meta = PromptMeta(prompt_id=prompt_id, prompt_vars=prompt_vars, prompt_hash=sha256_text(rendered))
        return RenderedPrompt(text=rendered, meta=meta)

