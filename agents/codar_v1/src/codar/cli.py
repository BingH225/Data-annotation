from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .agents.abductive_tot import AbductiveToTAgent
from .agents.conflict_engine import ConflictEngine
from .agents.critic import CriticAgent
from .agents.explicit_perception import ExplicitPerceptionAgent
from .agents.expectation import ExpectationAgent
from .agents.final_decision import FinalDecisionAgent
from .agents.null_hypothesis_gate import NullHypothesisGateAgent
from .agents.scenario_gate import ScenarioGateAgent
from .agents.social_context import SocialContextAgent
from .backends.factory import create_backend
from .config import load_config_bundle, validate_backend_config
from .eval.metrics import (
    build_nus_compat_detailed_records,
    evaluate_prediction_file,
    evaluate_prediction_file_nus_compat,
    load_ground_truth,
    load_predictions,
)
from .io.dataset import load_samples
from .logging_utils import RunLoggers
from .media import MediaResolver
from .orchestrator.baseline import DirectBaselineRunner
from .orchestrator.pipeline import CoDARPipeline
from .prompting import PromptStore
from .semantic_matcher import SemanticClosedSetMatcher
from .utils import utc_now_iso


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mk_pipeline(config_bundle, project_root: Path):
    runtime = config_bundle.runtime
    validate_backend_config(runtime)
    backend_cfg = runtime.get("backend", {})
    backend = create_backend(backend_cfg)
    prompt_store = PromptStore(project_root / "prompts")
    media_resolver = MediaResolver(runtime_cfg=runtime)
    max_stage_retries = int(runtime.get("pipeline", {}).get("max_stage_retries", 2))
    max_backtrack_rounds = int(runtime.get("pipeline", {}).get("max_backtrack_rounds", 2))
    max_video_frames = int(runtime.get("pipeline", {}).get("max_video_frames", 4))
    alpha_rule = float(runtime.get("pipeline", {}).get("alpha_rule", 0.6))
    alpha_llm = float(runtime.get("pipeline", {}).get("alpha_llm", 0.4))
    semantic_cfg = runtime.get("pipeline", {}).get("semantic_matcher", {}) or {}
    semantic_matcher = SemanticClosedSetMatcher(
        enabled=bool(semantic_cfg.get("enabled", True)),
        model_name=str(semantic_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        similarity_threshold=float(semantic_cfg.get("similarity_threshold", 0.35)),
    )
    anchor_rule = bool(runtime.get("pipeline", {}).get("enable_subject_anchor_rule", True))
    scenario_policy = config_bundle.scenario_policy
    pipeline = CoDARPipeline(
        scenario_gate=ScenarioGateAgent(backend, prompt_store, max_stage_retries),
        explicit_perception=ExplicitPerceptionAgent(backend, prompt_store, media_resolver, max_stage_retries, max_video_frames),
        social_context=SocialContextAgent(backend, prompt_store, max_stage_retries),
        expectation=ExpectationAgent(backend, prompt_store, max_stage_retries),
        conflict_engine=ConflictEngine(
            backend,
            prompt_store,
            max_stage_retries,
            alpha_rule=alpha_rule,
            alpha_llm=alpha_llm,
            scenario_policy=scenario_policy,
            semantic_matcher=semantic_matcher,
            semantic_threshold=float(semantic_cfg.get("rule_semantic_threshold", 0.45)),
        ),
        null_gate=NullHypothesisGateAgent(backend, prompt_store, max_stage_retries),
        abductive_tot=AbductiveToTAgent(backend, prompt_store, max_stage_retries),
        critic=CriticAgent(backend, prompt_store, max_stage_retries),
        final_decision=FinalDecisionAgent(
            backend,
            prompt_store,
            max_stage_retries,
            semantic_matcher=semantic_matcher,
            enable_subject_anchor_rule=anchor_rule,
        ),
        max_backtrack_rounds=max_backtrack_rounds,
    )
    return pipeline, backend


def _load_runtime(project_root: Path, config_path: Path, backend_override: str | None):
    bundle = load_config_bundle(project_root=project_root, runtime_path=config_path)
    if backend_override:
        bundle.runtime.setdefault("backend", {})["provider"] = backend_override
    return bundle


def _mk_baseline_runner(config_bundle, project_root: Path):
    runtime = config_bundle.runtime
    validate_backend_config(runtime)
    backend_cfg = runtime.get("backend", {})
    backend = create_backend(backend_cfg)
    prompt_store = PromptStore(project_root / "prompts")
    media_resolver = MediaResolver(runtime_cfg=runtime)
    max_stage_retries = int(runtime.get("pipeline", {}).get("max_stage_retries", 2))
    max_video_frames = int(runtime.get("pipeline", {}).get("max_video_frames", 4))
    baseline_temperature = float(runtime.get("baseline", {}).get("temperature", backend_cfg.get("temperature", 0.0)))
    runner = DirectBaselineRunner(
        backend=backend,
        prompt_store=prompt_store,
        media_resolver=media_resolver,
        max_stage_retries=max_stage_retries,
        max_video_frames=max_video_frames,
        temperature=baseline_temperature,
    )
    return runner, backend


def cmd_run_batch(args: argparse.Namespace) -> int:
    project_root = _project_root()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    bundle = _load_runtime(project_root, config_path, args.backend)
    pipeline, backend = _mk_pipeline(bundle, project_root)
    samples = load_samples(Path(args.input), scenario_filter=args.scenario, limit=args.limit)
    run_logs = RunLoggers(output_dir)
    run_logs.run_events.log(
        {
            "event": "run_start",
            "run_id": args.run_id or f"run_{utc_now_iso()}",
            "input": str(Path(args.input).resolve()),
            "count": len(samples),
            "backend": backend.metadata(),
            "scenario_filter": args.scenario,
        }
    )
    ok = 0
    failed = 0
    for idx, sample in enumerate(samples, start=1):
        result = pipeline.run_sample(sample=sample, backend_meta=backend.metadata())
        payload = asdict(result)
        run_logs.sample_records.log(payload)
        if result.error:
            failed += 1
            run_logs.failures.log(
                {
                    "event": "sample_failed",
                    "sample_id": result.sample_id,
                    "error": result.error,
                }
            )
        else:
            ok += 1
        if idx % 50 == 0:
            run_logs.run_events.log({"event": "progress", "processed": idx, "ok": ok, "failed": failed})
    summary = {"event": "run_complete", "total": len(samples), "ok": ok, "failed": failed}
    run_logs.run_events.log(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if failed == 0 else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    pred_path = Path(args.predictions).resolve()
    input_path = Path(args.input).resolve()
    output_metrics = Path(args.output_metrics).resolve()
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_prediction_file(pred_path, input_path)
    output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    nus_report, formula_report, formula_table = evaluate_prediction_file_nus_compat(pred_path, input_path)
    nus_path = output_metrics.with_name(f"{output_metrics.stem}.nus_compat.json")
    formula_path = output_metrics.with_name(f"{output_metrics.stem}.formula_metrics.json")
    formula_table_path = output_metrics.with_name(f"{output_metrics.stem}.formula_table_ready.json")
    detailed_path = (
        Path(args.output_detailed).resolve()
        if getattr(args, "output_detailed", None)
        else output_metrics.with_name(f"{output_metrics.stem}.detailed.json")
    )
    nus_path.write_text(json.dumps(nus_report, ensure_ascii=False, indent=2), encoding="utf-8")
    formula_path.write_text(json.dumps(formula_report, ensure_ascii=False, indent=2), encoding="utf-8")
    formula_table_path.write_text(json.dumps(formula_table, ensure_ascii=False, indent=2), encoding="utf-8")
    gt = load_ground_truth(input_path)
    preds = load_predictions(pred_path)
    detailed_records = build_nus_compat_detailed_records(gt, preds)
    detailed_path.write_text(json.dumps(detailed_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "basic_metrics_file": str(output_metrics),
                "nus_compat_metrics_file": str(nus_path),
                "formula_metrics_file": str(formula_path),
                "formula_table_file": str(formula_table_path),
                "detailed_file": str(detailed_path),
                "basic_metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    project_root = _project_root()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (project_root / "output" / "smoke")
    bundle = _load_runtime(project_root, config_path, args.backend)
    pipeline, backend = _mk_pipeline(bundle, project_root)
    samples = load_samples(Path(args.input), scenario_filter=args.scenario, limit=args.limit)
    if not samples:
        print("No samples for smoke.")
        return 1
    run_logs = RunLoggers(output_dir)
    run_logs.run_events.log({"event": "smoke_start", "count": len(samples), "backend": backend.metadata()})
    for sample in samples:
        result = pipeline.run_sample(sample=sample, backend_meta=backend.metadata())
        run_logs.sample_records.log(asdict(result))
        if result.error:
            run_logs.failures.log({"event": "smoke_sample_failed", "sample_id": result.sample_id, "error": result.error})
    run_logs.run_events.log({"event": "smoke_done"})
    print(f"Smoke completed for {len(samples)} sample(s). Output: {output_dir}")
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    project_root = _project_root()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    bundle = _load_runtime(project_root, config_path, args.backend)
    baseline_runner, backend = _mk_baseline_runner(bundle, project_root)
    samples = load_samples(Path(args.input), scenario_filter=args.scenario, limit=args.limit)
    run_logs = RunLoggers(output_dir)
    run_logs.run_events.log(
        {
            "event": "baseline_run_start",
            "run_id": args.run_id or f"baseline_{utc_now_iso()}",
            "input": str(Path(args.input).resolve()),
            "count": len(samples),
            "backend": backend.metadata(),
            "scenario_filter": args.scenario,
        }
    )
    ok = 0
    failed = 0
    for idx, sample in enumerate(samples, start=1):
        row = baseline_runner.run_sample(sample=sample, backend_meta=backend.metadata())
        run_logs.sample_records.log(row)
        if row.get("error"):
            failed += 1
            run_logs.failures.log(
                {
                    "event": "sample_failed",
                    "sample_id": row.get("sample_id"),
                    "error": row.get("error"),
                }
            )
        else:
            ok += 1
        if idx % 50 == 0:
            run_logs.run_events.log({"event": "progress", "processed": idx, "ok": ok, "failed": failed})
    summary = {"event": "baseline_run_complete", "total": len(samples), "ok": ok, "failed": failed}
    run_logs.run_events.log(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoDAR v1 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-batch", help="Run batch inference")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--output-dir", required=True)
    p_run.add_argument("--backend", choices=["vllm", "openai", "mock"], required=True)
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--scenario", choices=["affection", "attitude", "intent"])
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--run-id")
    p_run.set_defaults(func=cmd_run_batch)

    p_eval = sub.add_parser("evaluate", help="Evaluate predictions")
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--input", required=True)
    p_eval.add_argument("--output-metrics", required=True)
    p_eval.add_argument("--output-detailed")
    p_eval.set_defaults(func=cmd_evaluate)

    p_smoke = sub.add_parser("smoke", help="Run smoke test")
    p_smoke.add_argument("--input", required=True)
    p_smoke.add_argument("--scenario", choices=["affection", "attitude", "intent"], required=True)
    p_smoke.add_argument("--limit", type=int, default=3)
    p_smoke.add_argument("--backend", choices=["vllm", "openai", "mock"], required=True)
    p_smoke.add_argument("--config", required=True)
    p_smoke.add_argument("--output-dir")
    p_smoke.set_defaults(func=cmd_smoke)

    p_baseline = sub.add_parser("run-baseline", help="Run direct baseline (no agent stages)")
    p_baseline.add_argument("--input", required=True)
    p_baseline.add_argument("--output-dir", required=True)
    p_baseline.add_argument("--backend", choices=["vllm", "openai", "mock"], required=True)
    p_baseline.add_argument("--config", required=True)
    p_baseline.add_argument("--scenario", choices=["affection", "attitude", "intent"])
    p_baseline.add_argument("--limit", type=int)
    p_baseline.add_argument("--run-id")
    p_baseline.set_defaults(func=cmd_run_baseline)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
