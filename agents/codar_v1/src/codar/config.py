from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ConfigBundle:
    runtime: Dict[str, Any]
    scenario_policy: Dict[str, Any]
    thresholds: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be object: {path}")
    return data


def load_config_bundle(project_root: Path, runtime_path: Path) -> ConfigBundle:
    runtime = _load_yaml(runtime_path)
    scenario_policy = _load_yaml(project_root / "config" / "scenario_policy.yaml")
    thresholds = _load_yaml(project_root / "config" / "thresholds.yaml")
    return ConfigBundle(runtime=runtime, scenario_policy=scenario_policy, thresholds=thresholds)


def validate_backend_config(runtime: Dict[str, Any]) -> None:
    backend = runtime.get("backend", {})
    provider = backend.get("provider")
    if provider not in {"vllm", "openai", "mock"}:
        raise ValueError("backend.provider must be one of: vllm | openai | mock")
    if provider in {"vllm", "openai"}:
        required = ["model", "api_key", "base_url"]
        missing = []
        for k in required:
            v = str(backend.get(k, "")).strip()
            if not v or v.startswith("<FILL_"):
                missing.append(k)
        if missing:
            raise ValueError(
                "Missing backend placeholders. Fill runtime.yaml fields: "
                + ", ".join(f"backend.{k}" for k in missing)
            )
