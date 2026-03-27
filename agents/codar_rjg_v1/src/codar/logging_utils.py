from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .utils import utc_now_iso


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", utc_now_iso())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class RunLoggers:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_events = JsonlLogger(output_dir / "run_events.jsonl")
        self.sample_records = JsonlLogger(output_dir / "predictions.jsonl")
        self.failures = JsonlLogger(output_dir / "failures.jsonl")

