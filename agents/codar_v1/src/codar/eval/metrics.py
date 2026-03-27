from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def load_ground_truth(input_path: Path) -> Dict[str, Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    gt: Dict[str, Dict[str, Any]] = {}
    for row in data:
        sid = str(row.get("id", "")).strip()
        if not sid:
            continue
        scenario = str(_safe_get(row, "input", "scenario") or "").strip().lower()
        gt_row = row.get("ground_truth", {}) or {}
        diversity = row.get("diversity", {}) or {}
        gt[sid] = {
            "scenario": scenario,
            "subject": str(gt_row.get("subject", "")),
            "target": str(gt_row.get("target", "")),
            "mechanism": str(gt_row.get("mechanism", "")),
            "label": str(gt_row.get("label", "")),
            "domain": str(diversity.get("domain", "")),
            "culture": str(diversity.get("culture", "")),
        }
    return gt


def load_predictions(pred_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: Any) -> str:
    return str(text).strip().lower()


def normalize_taxonomy_value(text: Any) -> str:
    return " ".join(normalize_text(text).replace("_", " ").split())


def accuracy_score_local(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def macro_f1_score_local(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1_vals: List[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_vals.append(f1)
    return sum(f1_vals) / len(f1_vals)


def compute_metrics(gt_by_id: Dict[str, Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    correct = defaultdict(int)
    joint_correct = 0
    per_scenario = defaultdict(lambda: {"total": 0, "subject": 0, "target": 0, "mechanism": 0, "label": 0, "joint": 0})
    for row in predictions:
        sid = str(row.get("sample_id", ""))
        if sid not in gt_by_id:
            continue
        gt = gt_by_id[sid]
        pred = row.get("final_prediction", {}) or {}
        total += 1
        sc = gt["scenario"]
        per_scenario[sc]["total"] += 1
        flags = {}
        for key in ("subject", "target", "mechanism", "label"):
            ok = str(pred.get(key, "")) == str(gt.get(key, ""))
            flags[key] = ok
            if ok:
                correct[key] += 1
                per_scenario[sc][key] += 1
        if all(flags.values()):
            joint_correct += 1
            per_scenario[sc]["joint"] += 1

    metrics = {
        "total": total,
        "accuracy": {
            "subject": (correct["subject"] / total) if total else 0.0,
            "target": (correct["target"] / total) if total else 0.0,
            "mechanism": (correct["mechanism"] / total) if total else 0.0,
            "label": (correct["label"] / total) if total else 0.0,
            "joint": (joint_correct / total) if total else 0.0,
        },
        "per_scenario": {},
    }
    for sc, agg in per_scenario.items():
        t = agg["total"]
        metrics["per_scenario"][sc] = {
            "total": t,
            "subject_acc": (agg["subject"] / t) if t else 0.0,
            "target_acc": (agg["target"] / t) if t else 0.0,
            "mechanism_acc": (agg["mechanism"] / t) if t else 0.0,
            "label_acc": (agg["label"] / t) if t else 0.0,
            "joint_acc": (agg["joint"] / t) if t else 0.0,
        }
    return metrics


# === nus_evalfin_319.py compatible metrics ===
SCENARIOS = ["affection", "attitude", "intent"]
DOMAINS = [
    "Online & Social Media",
    "Public & Service",
    "Workplace",
    "Intimate Relationships",
    "Family Conversations",
    "Friend Group",
    "Education & Campus",
    "Friendship Interactions",
]
CULTURES = [
    "General Culture",
    "Arab Culture",
    "American Culture",
    "Muslim Culture",
    "African American Culture",
    "Jewish Culture",
    "Indian Culture",
    "East Asian Culture",
]

VALID_MECHANISMS = {
    "affection": ["multimodal incongruity", "figurative semantics", "affective deception", "socio_cultural dependency"],
    "intent": ["prosocial deception", "malicious manipulation", "expressive aggression", "benevolent provocation"],
    "attitude": ["dominant affiliation", "dominant detachment", "protective distancing", "submissive alignment"],
}
VALID_LABELS = {
    "affection": ["happy", "sad", "disgusted", "angry", "fearful", "bad"],
    "attitude": [
        "supportive",
        "appreciative",
        "sympathetic",
        "neutral",
        "indifferent",
        "concerned",
        "skeptical",
        "dismissive",
        "disapproving",
        "contemptuous",
        "hostile",
    ],
    "intent": ["mitigate", "intimidate", "alienate", "mock", "denounce", "provoke", "dominate", "condemn"],
}

FORMULA_FIELDS = ["label", "subject", "target", "mechanism"]


def _build_nus_compat_records(
    gt_by_id: Dict[str, Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in predictions:
        sid = str(row.get("sample_id") or row.get("id") or "").strip()
        if not sid or sid not in gt_by_id:
            continue
        gt = gt_by_id[sid]
        pred = row.get("final_prediction", {}) or row.get("prediction", {}) or {}

        matches = {
            "mechanism": normalize_taxonomy_value(pred.get("mechanism", "")) == normalize_taxonomy_value(gt.get("mechanism", "")),
            "label": normalize_taxonomy_value(pred.get("label", "")) == normalize_taxonomy_value(gt.get("label", "")),
            "subject": normalize_text(pred.get("subject", "")) == normalize_text(gt.get("subject", "")),
            "target": normalize_text(pred.get("target", "")) == normalize_text(gt.get("target", "")),
        }
        strict_match = all(matches.values())
        sc = normalize_text(gt.get("scenario", ""))
        pred_mech = normalize_taxonomy_value(pred.get("mechanism", ""))
        pred_label = normalize_taxonomy_value(pred.get("label", ""))
        allowed_mechs = [normalize_taxonomy_value(x) for x in VALID_MECHANISMS.get(sc, [])]
        allowed_labels = [normalize_taxonomy_value(x) for x in VALID_LABELS.get(sc, [])]
        records.append(
            {
                "id": sid,
                "ground_truth": {
                    "mechanism": gt.get("mechanism", ""),
                    "label": gt.get("label", ""),
                    "subject": gt.get("subject", ""),
                    "target": gt.get("target", ""),
                },
                "prediction": {
                    "mechanism": pred.get("mechanism", ""),
                    "label": pred.get("label", ""),
                    "subject": pred.get("subject", ""),
                    "target": pred.get("target", ""),
                },
                "matches": matches,
                "strict_match": strict_match,
                "meta_scenario": gt.get("scenario", ""),
                "meta_domain": gt.get("domain", ""),
                "meta_culture": gt.get("culture", ""),
                "error_analysis": {
                    "mech_mismatch": pred_mech not in allowed_mechs if allowed_mechs else True,
                    "label_mismatch": pred_label not in allowed_labels if allowed_labels else True,
                    # CoDAR outputs real entity text, not subject0/target0 slots.
                    "subject_format_error": False,
                    "target_format_error": False,
                    "predicted_mechanism": pred.get("mechanism", ""),
                    "predicted_label": pred.get("label", ""),
                    "predicted_subject": pred.get("subject", ""),
                    "predicted_target": pred.get("target", ""),
                },
            }
        )
    return records


def calculate_metrics_for_subset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"Count": 0}
    metrics: Dict[str, Any] = {"Count": len(records)}
    metrics["Strict_All4_Acc"] = accuracy_score_local([str(1)] * len(records), [str(int(bool(r.get("strict_match")))) for r in records])
    fields = ["mechanism", "label", "subject", "target"]
    for field in fields:
        y_true = [normalize_text(r.get("ground_truth", {}).get(field, "")) for r in records]
        y_pred = [normalize_text(r.get("prediction", {}).get(field, "")) for r in records]
        metrics[f"{field}_Accuracy"] = accuracy_score_local(y_true, y_pred)
        metrics[f"{field}_F1"] = macro_f1_score_local(y_true, y_pred)
    return metrics


def _formula_safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _formula_mean_optional(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _formula_get_scenario(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("meta_scenario", record.get("meta_situation", "")))


def _formula_get_match(record: Dict[str, Any], field: str) -> bool:
    matches = record.get("matches", {})
    if isinstance(matches, dict) and field in matches:
        return bool(matches[field])
    gt = normalize_text(record.get("ground_truth", {}).get(field, ""))
    pred = normalize_text(record.get("prediction", {}).get(field, ""))
    return gt == pred


def _formula_compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "N": 0,
            "independent_accuracy": {f: None for f in FORMULA_FIELDS},
            "conditional_accuracy": {"mechanism_given_label": None, "denominator_label_correct": 0},
            "relational_alignment": None,
            "joint_accuracy": None,
        }
    match_counts = {f: 0 for f in FORMULA_FIELDS}
    rel_correct = 0
    joint_correct = 0
    label_correct_count = 0
    mech_and_label_correct_count = 0
    for rec in records:
        row_match = {f: _formula_get_match(rec, f) for f in FORMULA_FIELDS}
        for f in FORMULA_FIELDS:
            if row_match[f]:
                match_counts[f] += 1
        if row_match["subject"] and row_match["target"]:
            rel_correct += 1
        if all(row_match.values()):
            joint_correct += 1
        if row_match["label"]:
            label_correct_count += 1
            if row_match["mechanism"]:
                mech_and_label_correct_count += 1
    return {
        "N": n,
        "independent_accuracy": {f: _formula_safe_div(match_counts[f], n) for f in FORMULA_FIELDS},
        "conditional_accuracy": {
            "mechanism_given_label": _formula_safe_div(mech_and_label_correct_count, label_correct_count),
            "denominator_label_correct": label_correct_count,
        },
        "relational_alignment": _formula_safe_div(rel_correct, n),
        "joint_accuracy": _formula_safe_div(joint_correct, n),
    }


def _formula_to_pct_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100.0, 2)


def _formula_to_table_metric_block(metrics: Dict[str, Any]) -> Dict[str, Any]:
    indep_parts = [
        metrics["independent_accuracy"].get("label"),
        metrics["independent_accuracy"].get("subject"),
        metrics["independent_accuracy"].get("target"),
        metrics["independent_accuracy"].get("mechanism"),
    ]
    indep = _formula_mean_optional(indep_parts)
    cond = metrics["conditional_accuracy"].get("mechanism_given_label")
    rel = metrics.get("relational_alignment")
    joint = metrics.get("joint_accuracy")
    return {
        "indep": indep,
        "cond": cond,
        "rel": rel,
        "joint": joint,
        "n": metrics.get("N", 0),
        "n_label_correct": metrics.get("conditional_accuracy", {}).get("denominator_label_correct", 0),
        "indep_components": metrics.get("independent_accuracy", {}),
    }


def _formula_build_table_ready(report: Dict[str, Any], scenario_order: List[str], method_name: str) -> Dict[str, Any]:
    order = scenario_order + ["overall"]
    raw_values: Dict[str, Dict[str, Any]] = {}
    pct_values: Dict[str, Dict[str, Any]] = {}
    for key in order:
        metrics = report["overall"] if key == "overall" else report["by_scenario"].get(key, _formula_compute_metrics([]))
        block = _formula_to_table_metric_block(metrics)
        raw_values[key] = block
        pct_values[key] = {
            "indep": _formula_to_pct_number(block["indep"]),
            "cond": _formula_to_pct_number(block["cond"]),
            "rel": _formula_to_pct_number(block["rel"]),
            "joint": _formula_to_pct_number(block["joint"]),
        }
    return {
        "method": method_name,
        "order": order,
        "columns": ["indep", "cond", "rel", "joint"],
        "values_raw": raw_values,
        "values_percent": pct_values,
    }


def build_nus_compat_metrics(
    gt_by_id: Dict[str, Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    method_name: str = "CoDAR-v1",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    records = _build_nus_compat_records(gt_by_id, predictions)
    overall = calculate_metrics_for_subset(records)
    by_scenario = {s: calculate_metrics_for_subset([r for r in records if normalize_text(r.get("meta_scenario", "")) == normalize_text(s)]) for s in SCENARIOS}
    by_domain = {d: calculate_metrics_for_subset([r for r in records if str(r.get("meta_domain", "")) == d]) for d in DOMAINS}
    by_culture = {c: calculate_metrics_for_subset([r for r in records if str(r.get("meta_culture", "")) == c]) for c in CULTURES}
    err_rep = {"mechanism_mismatches": {"count": 0, "samples": []}, "label_mismatches": {"count": 0, "samples": []}, "format_errors": {"count": 0, "samples": []}}
    for r in records:
        ea = r.get("error_analysis", {})
        if ea.get("mech_mismatch"):
            err_rep["mechanism_mismatches"]["count"] += 1
            err_rep["mechanism_mismatches"]["samples"].append({"id": r.get("id"), "given_sit": r.get("meta_scenario"), "pred_mech": ea.get("predicted_mechanism")})
        if ea.get("label_mismatch"):
            err_rep["label_mismatches"]["count"] += 1
            err_rep["label_mismatches"]["samples"].append({"id": r.get("id"), "given_sit": r.get("meta_scenario"), "pred_label": ea.get("predicted_label")})
        if ea.get("subject_format_error") or ea.get("target_format_error"):
            err_rep["format_errors"]["count"] += 1
            err_rep["format_errors"]["samples"].append({"id": r.get("id"), "pred_sub": ea.get("predicted_subject"), "pred_tgt": ea.get("predicted_target")})

    nus_report = {
        f"1. Overall ({len(records)} Successful Samples)": overall,
        "2. By Scenario": by_scenario,
        "3. By Domain": by_domain,
        "4. By Culture": by_culture,
        "5. Error Analysis": err_rep,
    }

    scenario_order = [normalize_text(x) for x in SCENARIOS]
    scenario_records = {s: [r for r in records if _formula_get_scenario(r) == s] for s in scenario_order}
    formula_report = {
        "total_records": len(records),
        "formula_note": {
            "independent_accuracy": "Acc_y for y in {Label, Subject, Target, Mechanism}",
            "table_indep": "Indep = mean(Acc_Label, Acc_Subject, Acc_Target, Acc_Mechanism)",
            "conditional_accuracy": "Acc_M|L = P(Mechanism correct | Label correct)",
            "relational_alignment": "Acc_S^T = P(Subject and Target both correct)",
            "joint_accuracy": "Acc_Joint = P(Label, Subject, Target, Mechanism all correct)",
        },
        "overall": _formula_compute_metrics(records),
        "by_scenario": {s: _formula_compute_metrics(scenario_records[s]) for s in scenario_order},
    }
    formula_table_ready = _formula_build_table_ready(formula_report, scenario_order, method_name)
    formula_report["table_ready"] = formula_table_ready
    return nus_report, formula_report, formula_table_ready


def build_nus_compat_detailed_records(
    gt_by_id: Dict[str, Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _build_nus_compat_records(gt_by_id, predictions)


def evaluate_prediction_file(predictions_path: Path, input_path: Path) -> Dict[str, Any]:
    gt = load_ground_truth(input_path)
    preds = load_predictions(predictions_path)
    return compute_metrics(gt, preds)


def evaluate_prediction_file_nus_compat(
    predictions_path: Path,
    input_path: Path,
    method_name: str = "CoDAR-v1",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    gt = load_ground_truth(input_path)
    preds = load_predictions(predictions_path)
    return build_nus_compat_metrics(gt, preds, method_name=method_name)
