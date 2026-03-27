from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

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
    compute_metrics,
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
from .rjg.fusion import RJGWeights, compute_total_score, default_rjg_weights, weights_to_dict
from .rjg.memory import build_memory_index, load_memory_index, save_memory_index
from .rjg.pipeline import RJGPipeline
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


def _mk_rjg_pipeline(config_bundle, project_root: Path, memory_path: Path):
    runtime = config_bundle.runtime
    validate_backend_config(runtime)
    backend_cfg = runtime.get("backend", {})
    backend = create_backend(backend_cfg)
    prompt_store = PromptStore(project_root / "prompts")
    media_resolver = MediaResolver(runtime_cfg=runtime)
    memory_index = load_memory_index(memory_path)
    max_stage_retries = int(runtime.get("pipeline", {}).get("max_stage_retries", 2))
    max_video_frames = int(runtime.get("pipeline", {}).get("max_video_frames", 4))
    rjg_cfg = runtime.get("rjg", {}) or {}
    base_w = default_rjg_weights()
    w_cfg = rjg_cfg.get("weights", {}) or {}
    weights = RJGWeights(
        retrieve_support=float(w_cfg.get("retrieve_support", base_w.retrieve_support)),
        judge_mech=float(w_cfg.get("judge_mech", base_w.judge_mech)),
        judge_label=float(w_cfg.get("judge_label", base_w.judge_label)),
        judge_role=float(w_cfg.get("judge_role", base_w.judge_role)),
        rule_cue=float(w_cfg.get("rule_cue", base_w.rule_cue)),
    )
    pipeline = RJGPipeline(
        backend=backend,
        prompt_store=prompt_store,
        media_resolver=media_resolver,
        memory_index=memory_index,
        scenario_policy=config_bundle.scenario_policy,
        max_retries=max_stage_retries,
        max_video_frames=max_video_frames,
        weights=weights,
        tie_margin=float(rjg_cfg.get("tie_margin", 0.06)),
        top_k=int(rjg_cfg.get("top_k", 40)),
        rerank_k=int(rjg_cfg.get("rerank_k", 12)),
        loo=bool(rjg_cfg.get("loo", True)),
    )
    return pipeline, backend, weights


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
            run_logs.failures.log({"event": "sample_failed", "sample_id": row.get("sample_id"), "error": row.get("error")})
        else:
            ok += 1
        if idx % 50 == 0:
            run_logs.run_events.log({"event": "progress", "processed": idx, "ok": ok, "failed": failed})
    summary = {"event": "baseline_run_complete", "total": len(samples), "ok": ok, "failed": failed}
    run_logs.run_events.log(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if failed == 0 else 1


def cmd_build_memory_index(args: argparse.Namespace) -> int:
    input_path = Path(args.input).resolve()
    output_path = Path(args.out).resolve()
    samples = load_samples(input_path, scenario_filter=args.scenario, limit=args.limit)
    index = build_memory_index(samples=samples, index_name="rjg_v1_unlabeled")
    save_memory_index(index, output_path)
    summary = {
        "event": "memory_index_built",
        "input": str(input_path),
        "output": str(output_path),
        "count": len(index.entries),
        "scenario_filter": args.scenario,
        "limit": args.limit,
        "loo": bool(args.loo),
    }
    report_path = output_path.with_suffix(".summary.json")
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_run_rjg_batch(args: argparse.Namespace) -> int:
    project_root = _project_root()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    memory_path = Path(args.memory).resolve()
    bundle = _load_runtime(project_root, config_path, args.backend)
    pipeline, backend, weights = _mk_rjg_pipeline(bundle, project_root, memory_path)
    samples = load_samples(Path(args.input), scenario_filter=args.scenario, limit=args.limit)
    run_logs = RunLoggers(output_dir)
    run_logs.run_events.log(
        {
            "event": "rjg_run_start",
            "run_id": args.run_id or f"rjg_{utc_now_iso()}",
            "input": str(Path(args.input).resolve()),
            "memory": str(memory_path),
            "count": len(samples),
            "backend": backend.metadata(),
            "weights": weights_to_dict(weights),
            "scenario_filter": args.scenario,
        }
    )
    ok = 0
    failed = 0
    for idx, sample in enumerate(samples, start=1):
        row = pipeline.run_sample(sample=sample, backend_meta=backend.metadata())
        run_logs.sample_records.log(row)
        if row.get("error"):
            failed += 1
            run_logs.failures.log({"event": "sample_failed", "sample_id": row.get("sample_id"), "error": row.get("error")})
        else:
            ok += 1
        if idx % 25 == 0:
            run_logs.run_events.log({"event": "progress", "processed": idx, "ok": ok, "failed": failed})
    summary = {"event": "rjg_run_complete", "total": len(samples), "ok": ok, "failed": failed}
    run_logs.run_events.log(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if failed == 0 else 1


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    vals = {k: max(0.0, float(v)) for k, v in raw.items()}
    total = sum(vals.values())
    if total <= 0:
        return weights_to_dict(default_rjg_weights())
    return {k: v / total for k, v in vals.items()}


def _decode_affection_dual_head_from_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {
            "subject": "",
            "target": "",
            "mechanism": "",
            "label": "",
            "confidence": 0.0,
            "score": -1e9,
        }

    def _mech_score(cand: Dict[str, Any]) -> float:
        comp = cand.get("components", {}) or {}
        return (
            1.0 * float(comp.get("judge_mech", 0.0))
            + 0.30 * float(comp.get("heuristic_agreement", 0.0))
            + 0.30 * float(comp.get("rule_cue", 0.0))
            - 0.40 * float(comp.get("penalty", 0.0))
        )

    mech_cand = max(candidates, key=lambda c: (_mech_score(c), float(c.get("total_score", 0.0))))
    label_votes: Dict[str, float] = {}
    for cand in candidates:
        comp = cand.get("components", {}) or {}
        label = str(cand.get("label", "")).strip()
        if not label:
            continue
        vote = (
            0.80 * float(comp.get("judge_label", 0.0))
            + 0.20 * float(comp.get("heuristic_agreement", 0.0))
            - 0.40 * float(comp.get("penalty", 0.0))
        )
        label_votes[label] = label_votes.get(label, 0.0) + vote
    if label_votes:
        best_label = max(label_votes.items(), key=lambda x: (x[1], x[0]))[0]
    else:
        best_label = str(mech_cand.get("label", ""))
    return {
        "subject": mech_cand.get("subject", ""),
        "target": mech_cand.get("target", ""),
        "mechanism": mech_cand.get("mechanism", ""),
        "label": best_label,
        "confidence": float(mech_cand.get("confidence", 0.0)),
        "score": float(mech_cand.get("total_score", 0.0)),
    }


def _trial_predictions_from_trace(rows: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    w = _normalize_weights(weights)
    weights_obj = RJGWeights(
        retrieve_support=w["retrieve_support"],
        judge_mech=w["judge_mech"],
        judge_label=w["judge_label"],
        judge_role=w["judge_role"],
        rule_cue=w["rule_cue"],
    )
    for row in rows:
        trace = row.get("trace", {}) or {}
        candidates = trace.get("candidate_scores", []) or []
        if not candidates:
            out.append(row)
            continue
        scenario = str(row.get("scenario", "")).strip().lower()
        if scenario == "affection" and False:
            dual = _decode_affection_dual_head_from_candidates(candidates)
            out.append(
                {
                    "sample_id": row.get("sample_id"),
                    "scenario": row.get("scenario"),
                    "final_prediction": {
                        "subject": dual.get("subject", ""),
                        "target": dual.get("target", ""),
                        "mechanism": dual.get("mechanism", ""),
                        "label": dual.get("label", ""),
                        "confidence": float(dual.get("confidence", 0.0)),
                        "decision_rationale_short": str(row.get("final_prediction", {}).get("decision_rationale_short", "")),
                    },
                    "stage_artifacts": row.get("stage_artifacts", []),
                    "backend_meta": row.get("backend_meta", {}),
                    "trace": {**trace, "tuned_weights": w, "tuned_total_score": float(dual.get("score", 0.0)), "tuned_decoder": "affection_dual_head"},
                    "error": row.get("error"),
                }
            )
            continue
        best = None
        best_score = -1e9
        for cand in candidates:
            comp = cand.get("components", {}) or {}
            score = compute_total_score(
                weights=weights_obj,
                retrieve_support=float(comp.get("retrieve_support", 0.0)),
                judge_mech=float(comp.get("judge_mech", 0.0)),
                judge_label=float(comp.get("judge_label", 0.0)),
                judge_role=float(comp.get("judge_role", 0.0)),
                rule_cue=float(comp.get("rule_cue", 0.0)),
                penalty=float(comp.get("penalty", 0.0)),
            )
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            out.append(row)
            continue
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "scenario": row.get("scenario"),
                "final_prediction": {
                    "subject": best.get("subject", ""),
                    "target": best.get("target", ""),
                    "mechanism": best.get("mechanism", ""),
                    "label": best.get("label", ""),
                    "confidence": float(best.get("confidence", 0.0)),
                    "decision_rationale_short": str(best.get("rationale", "")),
                },
                "stage_artifacts": row.get("stage_artifacts", []),
                "backend_meta": row.get("backend_meta", {}),
                "trace": {**trace, "tuned_weights": w, "tuned_total_score": best_score},
                "error": row.get("error"),
            }
        )
    return out


def _objective(metrics: Dict[str, Any], base_metrics: Dict[str, Any]) -> float:
    acc = metrics.get("accuracy", {}) or {}
    base = base_metrics.get("accuracy", {}) or {}
    mech = float(acc.get("mechanism", 0.0))
    label = float(acc.get("label", 0.0))
    joint = float(acc.get("joint", 0.0))
    score = 0.45 * mech + 0.45 * label + 0.10 * joint
    # Soft guardrail to prevent hurting role extraction.
    score -= max(0.0, float(base.get("subject", 0.0)) - float(acc.get("subject", 0.0))) * 0.15
    score -= max(0.0, float(base.get("target", 0.0)) - float(acc.get("target", 0.0))) * 0.15
    return score


def cmd_tune_rjg(args: argparse.Namespace) -> int:
    pred_path = Path(args.predictions).resolve()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_predictions(pred_path)
    gt = load_ground_truth(input_path)
    base_metrics = compute_metrics(gt, base_rows)

    grid = list(
        product(
            [0.25, 0.40, 0.55],
            [0.15, 0.25, 0.35],
            [0.15, 0.20, 0.30],
            [0.05, 0.10, 0.15],
            [0.00, 0.05, 0.10],
        )
    )
    random.seed(int(args.seed))
    random.shuffle(grid)
    budget = min(int(args.search_budget), len(grid))
    trials = grid[:budget]

    best_rows = base_rows
    best_metrics = base_metrics
    best_weights = weights_to_dict(default_rjg_weights())
    best_obj = _objective(base_metrics, base_metrics)
    history: List[Dict[str, Any]] = []
    for idx, (rs, jm, jl, jr, rc) in enumerate(trials, start=1):
        raw_w = {
            "retrieve_support": rs,
            "judge_mech": jm,
            "judge_label": jl,
            "judge_role": jr,
            "rule_cue": rc,
        }
        w = _normalize_weights(raw_w)
        trial_rows = _trial_predictions_from_trace(base_rows, w)
        metrics = compute_metrics(gt, trial_rows)
        obj = _objective(metrics, base_metrics)
        history.append({"trial": idx, "weights": w, "objective": obj, "metrics": metrics})
        if obj > best_obj:
            best_obj = obj
            best_weights = w
            best_rows = trial_rows
            best_metrics = metrics

    tuned_pred = output_dir / "predictions.tuned.jsonl"
    with tuned_pred.open("w", encoding="utf-8") as f:
        for row in best_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path = output_dir / "metrics.tuned.json"
    metrics_path.write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    best_path = output_dir / "best_weights.json"
    best_path.write_text(
        json.dumps(
            {
                "base_metrics": base_metrics,
                "best_metrics": best_metrics,
                "best_weights": best_weights,
                "objective": best_obj,
                "search_budget": budget,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path = output_dir / "tuning_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "predictions": str(tuned_pred),
                "metrics": str(metrics_path),
                "best_weights": str(best_path),
                "history": str(history_path),
                "best_metrics": best_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoDAR CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-batch", help="Run legacy CoDAR pipeline batch inference")
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

    p_smoke = sub.add_parser("smoke", help="Run smoke test on legacy CoDAR pipeline")
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

    p_mem = sub.add_parser("build-memory-index", help="Build unlabeled memory index from input samples")
    p_mem.add_argument("--input", required=True)
    p_mem.add_argument("--out", required=True)
    p_mem.add_argument("--scenario", choices=["affection", "attitude", "intent"])
    p_mem.add_argument("--limit", type=int)
    p_mem.add_argument("--loo", action="store_true", default=True)
    p_mem.set_defaults(func=cmd_build_memory_index)

    p_rjg = sub.add_parser("run-rjg-batch", help="Run RJG-v1 retrieval + judge graph inference")
    p_rjg.add_argument("--input", required=True)
    p_rjg.add_argument("--memory", required=True)
    p_rjg.add_argument("--output-dir", required=True)
    p_rjg.add_argument("--backend", choices=["vllm", "openai", "mock"], required=True)
    p_rjg.add_argument("--config", required=True)
    p_rjg.add_argument("--scenario", choices=["affection", "attitude", "intent"])
    p_rjg.add_argument("--limit", type=int)
    p_rjg.add_argument("--run-id")
    p_rjg.set_defaults(func=cmd_run_rjg_batch)

    p_tune = sub.add_parser("tune-rjg", help="Search RJG fusion weights using cached candidate traces")
    p_tune.add_argument("--predictions", required=True)
    p_tune.add_argument("--input", required=True)
    p_tune.add_argument("--output-dir", required=True)
    p_tune.add_argument("--search-budget", type=int, default=120)
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.set_defaults(func=cmd_tune_rjg)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
