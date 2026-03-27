import concurrent.futures
import copy
import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(r"D:\NUS\ACMm\Data-annotation")
TASK_DIR = BASE_DIR / "Task" / "05_adding_samples"
SOURCE_JSON_PATH = TASK_DIR / "Zuo_video_primary_day03.json"
CSV_PATH = TASK_DIR / "video_labels_Zuo_day03.csv"
EXPORT_STAGE2_PATH = TASK_DIR / "Zuo_video_primary_day03_export_stage2.json"
EXPORT_FINAL_PATH = TASK_DIR / "Zuo_video_primary_day03_export.json"
URL_REPORT_PATH = TASK_DIR / "Zuo_video_primary_day03_url_check_report.json"
FORMAT_REPORT_PATH = TASK_DIR / "Zuo_video_primary_day03_format_check_report.json"
REFERENCE_FORMAT_PATH = BASE_DIR / "export" / "All_exported_labels_new_format.json"

VIDEO_PREFIX = "https://huggingface.co/datasets/z4722/Implicit/blob/main/video/"
SCENARIOS = {"affection", "attitude", "intent"}
FORBIDDEN_CHARS = {"/", "\\", "(", ")"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "another",
    "other",
    "this",
    "that",
    "these",
    "those",
    "person",
    "individual",
}

GENERIC_SUBJECT_CANDIDATES = [
    "other coworker",
    "nearby customer",
    "store manager",
    "team member",
    "side bystander",
    "another friend",
    "group member",
    "local staff",
]

GENERIC_TARGET_CANDIDATES = [
    "the conversation",
    "the current topic",
    "store policy",
    "service process",
    "group discussion",
    "shared task",
    "background event",
    "other concern",
]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_ref(value: str) -> str:
    text = as_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return " ".join(tokens)


def scenario_from_value(value: str) -> str:
    s = as_text(value).lower()
    if s in SCENARIOS:
        return s
    return s


def sanitize_option(value: str) -> str:
    text = as_text(value)
    for ch in FORBIDDEN_CHARS:
        text = text.replace(ch, " ")
    text = re.sub(r"[\[\]{}<>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_option_format(value: str) -> bool:
    text = sanitize_option(value)
    if not text:
        return False
    if any(ch in text for ch in FORBIDDEN_CHARS):
        return False
    if len(text.split()) > 5:
        return False
    return True


def refers_same(gt: str, cand: str) -> bool:
    g = normalize_ref(gt)
    c = normalize_ref(cand)
    if not g or not c:
        return False
    if g == c:
        return True

    g_tokens = g.split()
    c_tokens = c.split()

    if len(g_tokens) == 1 and g_tokens[0] in c_tokens:
        return True
    if len(c_tokens) == 1 and c_tokens[0] in g_tokens:
        return True

    pronoun_groups = [
        {"he", "him", "his", "man", "male", "guy", "boy"},
        {"she", "her", "hers", "woman", "female", "girl", "lady"},
        {"they", "them", "their"},
    ]
    for group in pronoun_groups:
        if g in group and c in group:
            return True

    return False


def extract_suffix_from_local_path(local_path: str) -> str:
    norm = as_text(local_path).replace("\\", "/")
    lower = norm.lower()
    marker = "rawdata/"
    idx = lower.find(marker)
    if idx != -1:
        return norm[idx + len(marker) :].lstrip("/")
    raise ValueError(f"Cannot parse rawdata suffix from media_path_local: {local_path}")


def check_url_exists(url: str, timeout: int = 20) -> Tuple[str, bool, Optional[int], str]:
    req_head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req_head, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            return url, status == 200, int(status), ""
    except urllib.error.HTTPError as e:
        if e.code in (403, 405):
            # Some endpoints do not support HEAD; retry with GET.
            pass
        else:
            return url, False, int(e.code), str(e)
    except Exception as e:  # pragma: no cover
        # Retry with GET below
        _ = e

    req_get = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req_get, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            return url, status == 200, int(status), ""
    except urllib.error.HTTPError as e:
        return url, False, int(e.code), str(e)
    except Exception as e:
        return url, False, None, str(e)


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def pick_scenario(record: Dict[str, Any]) -> str:
    out = record.get("output", {}) if isinstance(record.get("output"), dict) else {}
    inp = record.get("input", {}) if isinstance(record.get("input"), dict) else {}
    candidates = [out.get("situation"), out.get("scenario"), inp.get("scenario")]
    for value in candidates:
        s = scenario_from_value(value)
        if s in SCENARIOS:
            return s
    return scenario_from_value(candidates[0])


def pick_mechanism(out: Dict[str, Any], scenario: str) -> str:
    generic = as_text(out.get("mechanism"))
    if generic and generic.upper() != "NULL":
        return generic
    key = f"mechanism_{scenario.capitalize()}"
    specific = as_text(out.get(key))
    if specific and specific.upper() != "NULL":
        return specific
    for suffix in ("Affection", "Attitude", "Intent"):
        cross = as_text(out.get(f"mechanism_{suffix}"))
        if cross and cross.upper() != "NULL":
            return cross
    return ""


def pick_label(out: Dict[str, Any], scenario: str) -> str:
    generic = as_text(out.get("label"))
    if generic and generic.upper() != "NULL":
        return generic
    key = f"label_{scenario.capitalize()}"
    specific = as_text(out.get(key))
    if specific and specific.upper() != "NULL":
        return specific
    for suffix in ("Affection", "Attitude", "Intent"):
        cross = as_text(out.get(f"label_{suffix}"))
        if cross and cross.upper() != "NULL":
            return cross
    return ""


def build_candidate_pools(reference_items: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    pools: Dict[Tuple[str, str], List[str]] = {}
    seen: Dict[Tuple[str, str], set] = {}

    for item in reference_items:
        inp = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
        scenario = scenario_from_value(inp.get("scenario"))
        options = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
        gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}

        for field in ("subject", "target"):
            values: List[str] = []
            opt_values = options.get(field)
            if isinstance(opt_values, list):
                values.extend([as_text(x) for x in opt_values])
            values.append(as_text(gt.get(field)))

            for raw in values:
                cand = sanitize_option(raw)
                if not is_valid_option_format(cand):
                    continue
                key = normalize_ref(cand)
                if not key:
                    continue
                for bucket in ((scenario, field), ("all", field)):
                    if bucket not in pools:
                        pools[bucket] = []
                        seen[bucket] = set()
                    if key in seen[bucket]:
                        continue
                    seen[bucket].add(key)
                    pools[bucket].append(cand)
    return pools


def context_candidates(record: Dict[str, Any], field: str) -> List[str]:
    inp = record.get("input", {}) if isinstance(record.get("input"), dict) else {}
    text = as_text(inp.get("text"))
    audio = as_text(inp.get("audio_caption"))
    local_path = as_text(inp.get("media_path_local")).lower()
    lower = f"{text} {audio}".lower()

    cands: List[str] = []
    if field == "subject":
        if "store" in lower:
            cands.extend(["store manager", "store employee", "nearby customer"])
        if "friend" in lower or "party" in lower:
            cands.extend(["close friend", "party guest"])
        if "speaker" in lower:
            cands.extend(["the listener", "nearby bystander"])
        if "mustard" in local_path:
            cands.append("sitcom character")
        if "mintrec" in local_path:
            cands.append("video participant")
        cands.extend(GENERIC_SUBJECT_CANDIDATES)
    else:
        if "store" in lower:
            cands.extend(["store policy", "checkout process"])
        if "phone" in lower:
            cands.append("the mobile phone")
        if "meeting" in lower or "work" in lower:
            cands.extend(["work schedule", "team task"])
        if "mustard" in local_path:
            cands.append("another dialogue topic")
        if "mintrec" in local_path:
            cands.append("current social context")
        cands.extend(GENERIC_TARGET_CANDIDATES)

    names = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    for name in names[:3]:
        if field == "subject":
            cands.append(name)
        else:
            cands.append(f"about {name}")
    return [sanitize_option(x) for x in cands if sanitize_option(x)]


def choose_distractors(
    ground_truth: str,
    field: str,
    scenario: str,
    record: Dict[str, Any],
    pools: Dict[Tuple[str, str], List[str]],
) -> List[str]:
    gt = sanitize_option(ground_truth)
    used_keys = {normalize_ref(gt)} if normalize_ref(gt) else set()

    candidates: List[str] = []
    candidates.extend(context_candidates(record, field))
    candidates.extend(pools.get((scenario, field), []))
    candidates.extend(pools.get(("all", field), []))
    candidates.extend(GENERIC_SUBJECT_CANDIDATES if field == "subject" else GENERIC_TARGET_CANDIDATES)

    distractors: List[str] = []
    for cand_raw in candidates:
        cand = sanitize_option(cand_raw)
        if not is_valid_option_format(cand):
            continue
        key = normalize_ref(cand)
        if not key:
            continue
        if key in used_keys:
            continue
        if refers_same(gt, cand):
            continue
        distractors.append(cand)
        used_keys.add(key)
        if len(distractors) == 3:
            return distractors

    # Deterministic fallback to guarantee 3 distractors.
    idx = 1
    while len(distractors) < 3:
        fallback = sanitize_option(f"other {field} {idx}")
        key = normalize_ref(fallback)
        if key and key not in used_keys and not refers_same(gt, fallback):
            distractors.append(fallback)
            used_keys.add(key)
        idx += 1
    return distractors


def build_new_format_record(record: Dict[str, Any], pools: Dict[Tuple[str, str], List[str]]) -> Dict[str, Any]:
    inp = record.get("input", {}) if isinstance(record.get("input"), dict) else {}
    out = record.get("output", {}) if isinstance(record.get("output"), dict) else {}

    scenario = pick_scenario(record)
    sample_id = as_text(inp.get("id") or inp.get("samples_id"))

    gt_subject = sanitize_option(as_text(out.get("subject")))
    gt_target = sanitize_option(as_text(out.get("target")))

    subj_d = choose_distractors(gt_subject, "subject", scenario, record, pools)
    targ_d = choose_distractors(gt_target, "target", scenario, record, pools)

    return {
        "id": sample_id,
        "input": {
            "scenario": scenario,
            "text": as_text(inp.get("text")),
            "media": {
                "video_url": as_text(inp.get("media_path")),
                "audio_url": as_text(inp.get("audio_path")),
                "audio_caption": as_text(inp.get("audio_caption")),
                "video_path": as_text(inp.get("media_path_local")),
                "audio_path": as_text(inp.get("audio_path_local")),
            },
        },
        "options": {
            "subject": [gt_subject] + subj_d,
            "target": [gt_target] + targ_d,
        },
        "ground_truth": {
            "subject": gt_subject,
            "target": gt_target,
            "mechanism": pick_mechanism(out, scenario),
            "label": pick_label(out, scenario),
        },
        "diversity": {
            "domain": as_text(out.get("domain")),
            "culture": as_text(out.get("culture")),
        },
    }


def validate_new_format(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        sid = as_text(item.get("id"))
        options = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
        gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}

        for field in ("subject", "target"):
            vals = options.get(field, [])
            if not isinstance(vals, list) or len(vals) != 4:
                issues.append({"index": i, "id": sid, "field": field, "issue": "options_not_len4"})
                continue

            gt_val = as_text(gt.get(field))
            gt_key = normalize_ref(gt_val)
            keys = [normalize_ref(as_text(v)) for v in vals]

            # Exactly one option aligned with ground truth referent.
            gt_hits = sum(1 for k in keys if k and k == gt_key)
            if gt_hits != 1:
                issues.append({
                    "index": i,
                    "id": sid,
                    "field": field,
                    "issue": "gt_referent_not_exactly_once",
                    "values": vals,
                    "ground_truth": gt_val,
                })

            if len(set(keys)) != 4:
                issues.append({
                    "index": i,
                    "id": sid,
                    "field": field,
                    "issue": "duplicate_referent",
                    "values": vals,
                })

            for j, val in enumerate(vals):
                text = as_text(val)
                if j == 0:
                    continue
                if not is_valid_option_format(text):
                    issues.append({
                        "index": i,
                        "id": sid,
                        "field": field,
                        "issue": "invalid_distractor_format",
                        "value": text,
                    })
                if refers_same(gt_val, text):
                    issues.append({
                        "index": i,
                        "id": sid,
                        "field": field,
                        "issue": "distractor_same_referent_as_gt",
                        "value": text,
                        "ground_truth": gt_val,
                    })
    return issues


def main() -> None:
    source_records = json.loads(SOURCE_JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(source_records, list):
        raise ValueError("Source JSON must be a list.")

    # Step 1: update media_path and check URL existence.
    updated_source = copy.deepcopy(source_records)
    check_tasks: List[Tuple[str, str]] = []  # (id, url)
    for rec in updated_source:
        inp = rec.get("input", {}) if isinstance(rec.get("input"), dict) else {}
        suffix = extract_suffix_from_local_path(as_text(inp.get("media_path_local")))
        new_url = VIDEO_PREFIX + suffix
        inp["media_path"] = new_url
        rec["input"] = inp
        check_tasks.append((as_text(inp.get("id") or inp.get("samples_id")), new_url))

    backup_path = SOURCE_JSON_PATH.with_suffix(".json.bak_before_url_update")
    if not backup_path.exists():
        backup_path.write_text(json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8")

    SOURCE_JSON_PATH.write_text(json.dumps(updated_source, ensure_ascii=False, indent=2), encoding="utf-8")

    url_results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        future_map = {
            ex.submit(check_url_exists, url): (sid, url)
            for sid, url in check_tasks
        }
        for fut in concurrent.futures.as_completed(future_map):
            sid, url = future_map[fut]
            checked_url, exists, status, error = fut.result()
            url_results.append(
                {
                    "id": sid,
                    "url": checked_url,
                    "exists": exists,
                    "status_code": status,
                    "error": error,
                }
            )

    missing = [x for x in url_results if not x["exists"]]
    URL_REPORT_PATH.write_text(
        json.dumps(
            {
                "total": len(url_results),
                "missing_count": len(missing),
                "missing": missing,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 2: overwrite values from CSV into exported stage2 JSON.
    csv_rows = load_csv_rows(CSV_PATH)
    if len(csv_rows) != len(updated_source):
        raise ValueError(f"CSV rows ({len(csv_rows)}) != JSON rows ({len(updated_source)}).")

    row_by_id = {as_text(r.get("id")): r for r in csv_rows if as_text(r.get("id"))}

    stage2_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(updated_source):
        rec2 = copy.deepcopy(rec)
        inp = rec2.get("input", {}) if isinstance(rec2.get("input"), dict) else {}
        out = rec2.get("output", {}) if isinstance(rec2.get("output"), dict) else {}

        sid = as_text(inp.get("id") or inp.get("samples_id"))
        row = row_by_id.get(sid, csv_rows[idx])

        inp["text"] = as_text(row.get("input_text"))
        inp["audio_caption"] = as_text(row.get("audio_caption"))

        out["subject"] = as_text(row.get("subject"))
        out["target"] = as_text(row.get("target"))
        out["situation"] = as_text(row.get("situation"))
        out["mechanism_Affection"] = as_text(row.get("mechanism_Affection"))
        out["mechanism_Intent"] = as_text(row.get("mechanism_Intent"))
        out["mechanism_Attitude"] = as_text(row.get("mechanism_Attitude"))
        out["mechanism"] = as_text(row.get("mechanism"))
        out["domain"] = as_text(row.get("domain"))
        out["culture"] = as_text(row.get("culture"))
        out["label_Affection"] = as_text(row.get("label_Affection"))
        out["label_Intent"] = as_text(row.get("label_Intent"))
        out["label_Attitude"] = as_text(row.get("label_Attitude"))
        out["label"] = pick_label(out, scenario_from_value(out.get("situation")))
        out["rationale"] = as_text(row.get("rationale"))

        rec2["input"] = inp
        rec2["output"] = out
        stage2_records.append(rec2)

    EXPORT_STAGE2_PATH.write_text(
        json.dumps(stage2_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Step 3 + 4: convert to new format and generate distractors.
    reference_items = json.loads(REFERENCE_FORMAT_PATH.read_text(encoding="utf-8"))
    if not isinstance(reference_items, list):
        raise ValueError("Reference format JSON must be a list.")

    pools = build_candidate_pools(reference_items)
    final_items = [build_new_format_record(rec, pools) for rec in stage2_records]

    issues = validate_new_format(final_items)
    FORMAT_REPORT_PATH.write_text(
        json.dumps(
            {
                "total": len(final_items),
                "issue_count": len(issues),
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    EXPORT_FINAL_PATH.write_text(
        json.dumps(final_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Updated source JSON: {SOURCE_JSON_PATH}")
    print(f"URL check report: {URL_REPORT_PATH}")
    print(f"Stage2 export: {EXPORT_STAGE2_PATH}")
    print(f"Final export: {EXPORT_FINAL_PATH}")
    print(f"Format check report: {FORMAT_REPORT_PATH}")
    print(f"URL missing count: {len(missing)}")
    print(f"Final format issue count: {len(issues)}")


if __name__ == "__main__":
    main()
