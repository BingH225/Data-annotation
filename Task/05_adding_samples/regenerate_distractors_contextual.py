import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(r"D:\NUS\ACMm\Data-annotation")
TASK_DIR = BASE_DIR / "Task" / "05_adding_samples"
STAGE2_PATH = TASK_DIR / "Zuo_video_primary_day03_export_stage2.json"
FINAL_PATH = TASK_DIR / "Zuo_video_primary_day03_export.json"
REFERENCE_PATH = BASE_DIR / "export" / "All_exported_labels_new_format.json"
CHECK_REPORT_PATH = TASK_DIR / "Zuo_video_primary_day03_format_check_report.json"
DISTRACTOR_REPORT_PATH = TASK_DIR / "Zuo_video_primary_day03_distractor_generation_report.json"

FORBIDDEN_CHARS = {"/", "\\", "(", ")"}
SCENARIOS = {"affection", "attitude", "intent"}
PLACEHOLDER_MARKERS = {
    "use non",
    "distractor",
    "placeholder",
    "other subject",
    "other target",
    "alternative subject",
    "alternative target",
}
GENERIC_BANNED = {
    "the conversation",
    "the current topic",
    "group discussion",
    "shared task",
    "background event",
    "other concern",
}

STOPWORDS = {
    "a","an","the","this","that","these","those","is","are","was","were","be","been","being",
    "i","you","he","she","it","they","we","me","him","her","them","my","your","his","their","our",
    "to","of","in","on","at","for","with","about","and","or","but","if","then","than","as","by",
    "from","up","down","out","into","over","under","again","further","once","just","so","very",
    "can","could","would","should","will","shall","do","does","did","have","has","had",
    "not","no","yes","oh","uh","um","yeah","okay","ok",
}

ROLE_HINTS_SUBJECT = [
    "store manager",
    "store employee",
    "customer",
    "coworker",
    "friend",
    "guest",
    "driver",
    "cashier",
    "waiter",
    "host",
    "teammate",
    "speaker",
    "listener",
]

ROLE_HINTS_TARGET = [
    "store policy",
    "work schedule",
    "payment process",
    "friendship issue",
    "party plan",
    "service request",
    "delivery delay",
    "team decision",
    "customer complaint",
    "ongoing argument",
]


def as_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def sanitize_phrase(text: str) -> str:
    s = as_text(text)
    for ch in FORBIDDEN_CHARS:
        s = s.replace(ch, " ")
    s = re.sub(r"[\[\]{}<>|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clamp_words(text: str, max_words: int = 5) -> str:
    words = sanitize_phrase(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def is_valid_option(text: str) -> bool:
    s = sanitize_phrase(text)
    if not s:
        return False
    if any(ch in s for ch in FORBIDDEN_CHARS):
        return False
    if len(s.split()) > 5:
        return False
    return True


def norm_ref(text: str) -> str:
    s = sanitize_phrase(text).lower()
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(toks)


def same_referent(a: str, b: str) -> bool:
    na = norm_ref(a)
    nb = norm_ref(b)
    if not na or not nb:
        return False
    if na == nb:
        return True

    ta = set(na.split())
    tb = set(nb.split())
    if len(ta) == 1 and ta.issubset(tb):
        return True
    if len(tb) == 1 and tb.issubset(ta):
        return True

    male = {"he","him","his","man","male","guy","boy"}
    female = {"she","her","hers","woman","female","girl","lady"}
    if na in male and nb in male:
        return True
    if na in female and nb in female:
        return True
    return False


def tokenize_context(text: str) -> Set[str]:
    s = sanitize_phrase(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS and len(t) > 2]
    return set(toks)


def scenario_of_stage2(rec: Dict[str, Any]) -> str:
    out = rec.get("output", {}) if isinstance(rec.get("output"), dict) else {}
    inp = rec.get("input", {}) if isinstance(rec.get("input"), dict) else {}
    for v in (out.get("situation"), out.get("scenario"), inp.get("scenario")):
        s = as_text(v).lower()
        if s in SCENARIOS:
            return s
    return as_text(out.get("situation")).lower()


def mechanism_from_out(out: Dict[str, Any], scenario: str) -> str:
    generic = as_text(out.get("mechanism"))
    if generic and generic.upper() != "NULL":
        return generic
    key = f"mechanism_{scenario.capitalize()}"
    v = as_text(out.get(key))
    if v and v.upper() != "NULL":
        return v
    for suf in ("Affection","Attitude","Intent"):
        vv = as_text(out.get(f"mechanism_{suf}"))
        if vv and vv.upper() != "NULL":
            return vv
    return ""


def label_from_out(out: Dict[str, Any], scenario: str) -> str:
    generic = as_text(out.get("label"))
    if generic and generic.upper() != "NULL":
        return generic
    key = f"label_{scenario.capitalize()}"
    v = as_text(out.get(key))
    if v and v.upper() != "NULL":
        return v
    for suf in ("Affection","Attitude","Intent"):
        vv = as_text(out.get(f"label_{suf}"))
        if vv and vv.upper() != "NULL":
            return vv
    return ""


def is_bad_candidate(text: str) -> bool:
    s = sanitize_phrase(text).lower()
    if not s:
        return True
    if s in GENERIC_BANNED:
        return True
    for m in PLACEHOLDER_MARKERS:
        if m in s:
            return True
    return False


def build_reference_index(reference: List[Dict[str, Any]]):
    # entry: scenario, context_tokens, subject_options, target_options
    entries = []
    scenario_pool = defaultdict(lambda: defaultdict(list))
    seen_pool = defaultdict(lambda: defaultdict(set))

    for item in reference:
        inp = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
        media = inp.get("media", {}) if isinstance(inp.get("media"), dict) else {}
        gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}
        opts = item.get("options", {}) if isinstance(item.get("options"), dict) else {}

        scenario = as_text(inp.get("scenario")).lower()
        ctx = " ".join([
            as_text(inp.get("text")),
            as_text(media.get("audio_caption")),
            as_text(gt.get("subject")),
            as_text(gt.get("target")),
        ])
        ctx_tokens = tokenize_context(ctx)

        sub_vals = []
        tar_vals = []
        if isinstance(opts.get("subject"), list):
            sub_vals.extend([as_text(x) for x in opts["subject"]])
        sub_vals.append(as_text(gt.get("subject")))
        if isinstance(opts.get("target"), list):
            tar_vals.extend([as_text(x) for x in opts["target"]])
        tar_vals.append(as_text(gt.get("target")))

        def clean_vals(vals):
            out_vals = []
            seen = set()
            for raw in vals:
                cand = clamp_words(raw)
                key = norm_ref(cand)
                if not cand or not key:
                    continue
                if is_bad_candidate(cand):
                    continue
                if not is_valid_option(cand):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                out_vals.append(cand)
            return out_vals

        sub_clean = clean_vals(sub_vals)
        tar_clean = clean_vals(tar_vals)

        entries.append(
            {
                "scenario": scenario,
                "tokens": ctx_tokens,
                "subject": sub_clean,
                "target": tar_clean,
            }
        )

        for field, values in (("subject", sub_clean), ("target", tar_clean)):
            for v in values:
                key = norm_ref(v)
                for bucket in (scenario, "all"):
                    if key in seen_pool[bucket][field]:
                        continue
                    seen_pool[bucket][field].add(key)
                    scenario_pool[bucket][field].append(v)

    return entries, scenario_pool


def similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    if inter == 0:
        return 0.0
    return inter / (len(a) ** 0.5 * len(b) ** 0.5)


def context_fallback_candidates(rec: Dict[str, Any], field: str) -> List[str]:
    inp = rec.get("input", {}) if isinstance(rec.get("input"), dict) else {}
    out = rec.get("output", {}) if isinstance(rec.get("output"), dict) else {}
    txt = " ".join([
        as_text(inp.get("text")),
        as_text(inp.get("audio_caption")),
        as_text(out.get("rationale")),
    ])

    candidates = []

    names = re.findall(r"\b[A-Z][a-z]{2,}\b", txt)
    for n in names[:8]:
        candidates.append(n)

    low = txt.lower()
    if field == "subject":
        for role in ROLE_HINTS_SUBJECT:
            key = role.split()[-1]
            if key in low or role in low:
                candidates.append(role)
        candidates.extend(ROLE_HINTS_SUBJECT)
    else:
        for role in ROLE_HINTS_TARGET:
            key = role.split()[-1]
            if key in low or role in low:
                candidates.append(role)
        # derive from salient nouns in text/audio/rationale
        toks = [t for t in tokenize_context(txt) if len(t) >= 4]
        for t in toks[:15]:
            candidates.append(f"the {t}")
        candidates.extend(ROLE_HINTS_TARGET)

    out_vals = []
    seen = set()
    for c in candidates:
        s = clamp_words(c)
        k = norm_ref(s)
        if not s or not k:
            continue
        if is_bad_candidate(s):
            continue
        if not is_valid_option(s):
            continue
        if k in seen:
            continue
        seen.add(k)
        out_vals.append(s)
    return out_vals


def pick_three_distractors(
    gt: str,
    field: str,
    scenario: str,
    rec: Dict[str, Any],
    ref_entries: List[Dict[str, Any]],
    scenario_pool,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    gt_clean = clamp_words(gt)
    gt_key = norm_ref(gt_clean)

    inp = rec.get("input", {}) if isinstance(rec.get("input"), dict) else {}
    out = rec.get("output", {}) if isinstance(rec.get("output"), dict) else {}
    q_tokens = tokenize_context(
        " ".join([
            as_text(inp.get("text")),
            as_text(inp.get("audio_caption")),
            as_text(out.get("rationale")),
            gt_clean,
        ])
    )

    scored_refs = []
    for idx, e in enumerate(ref_entries):
        score = similarity(q_tokens, e["tokens"])
        if e["scenario"] == scenario:
            score += 0.08
        if score <= 0:
            continue
        scored_refs.append((score, idx))
    scored_refs.sort(reverse=True)

    used_keys = {gt_key} if gt_key else set()
    chosen: List[str] = []
    trace: List[Dict[str, Any]] = []

    def try_add(candidate: str, source: str, score: float):
        nonlocal chosen
        c = clamp_words(candidate)
        k = norm_ref(c)
        if not c or not k:
            return
        if is_bad_candidate(c):
            return
        if not is_valid_option(c):
            return
        if k in used_keys:
            return
        if same_referent(gt_clean, c):
            return
        used_keys.add(k)
        chosen.append(c)
        trace.append({"value": c, "source": source, "score": round(score, 4)})

    # 1) Top similar reference entries, same field options.
    for score, ridx in scored_refs[:160]:
        if len(chosen) >= 3:
            break
        e = ref_entries[ridx]
        for c in e[field]:
            if len(chosen) >= 3:
                break
            try_add(c, f"retrieval:{ridx}", score)

    # 2) Scenario-level pool.
    for c in scenario_pool.get(scenario, {}).get(field, []):
        if len(chosen) >= 3:
            break
        try_add(c, "scenario_pool", 0.0)

    # 3) Global pool.
    for c in scenario_pool.get("all", {}).get(field, []):
        if len(chosen) >= 3:
            break
        try_add(c, "global_pool", 0.0)

    # 4) Context fallback (non-template, derived from record text/rationale).
    for c in context_fallback_candidates(rec, field):
        if len(chosen) >= 3:
            break
        try_add(c, "context_fallback", 0.0)

    # 5) Safety deterministic fallback.
    idx = 1
    while len(chosen) < 3:
        c = clamp_words(f"alt {field} {idx}")
        try_add(c, "safety_fallback", 0.0)
        idx += 1

    return chosen, trace


def build_record(rec: Dict[str, Any], ref_entries, scenario_pool):
    inp = rec.get("input", {}) if isinstance(rec.get("input"), dict) else {}
    out = rec.get("output", {}) if isinstance(rec.get("output"), dict) else {}

    scenario = scenario_of_stage2(rec)
    sid = as_text(inp.get("id") or inp.get("samples_id"))

    gt_subject = clamp_words(as_text(out.get("subject")))
    gt_target = clamp_words(as_text(out.get("target")))

    sub_d, sub_trace = pick_three_distractors(
        gt_subject, "subject", scenario, rec, ref_entries, scenario_pool
    )
    tar_d, tar_trace = pick_three_distractors(
        gt_target, "target", scenario, rec, ref_entries, scenario_pool
    )

    item = {
        "id": sid,
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
            "subject": [gt_subject] + sub_d,
            "target": [gt_target] + tar_d,
        },
        "ground_truth": {
            "subject": gt_subject,
            "target": gt_target,
            "mechanism": mechanism_from_out(out, scenario),
            "label": label_from_out(out, scenario),
        },
        "diversity": {
            "domain": as_text(out.get("domain")),
            "culture": as_text(out.get("culture")),
        },
    }

    trace = {
        "id": sid,
        "scenario": scenario,
        "subject_trace": sub_trace,
        "target_trace": tar_trace,
    }
    return item, trace


def validate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    for i, item in enumerate(items):
        sid = as_text(item.get("id"))
        opts = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
        gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}

        for field in ("subject", "target"):
            vals = opts.get(field, [])
            if not isinstance(vals, list) or len(vals) != 4:
                issues.append({"index": i, "id": sid, "field": field, "issue": "len_not_4"})
                continue

            gtv = as_text(gt.get(field))
            gt_key = norm_ref(gtv)
            keys = [norm_ref(as_text(v)) for v in vals]

            if sum(1 for k in keys if k == gt_key and k) != 1:
                issues.append({
                    "index": i,
                    "id": sid,
                    "field": field,
                    "issue": "gt_ref_not_exactly_one",
                    "values": vals,
                    "ground_truth": gtv,
                })

            if len(set(keys)) != 4:
                issues.append({
                    "index": i,
                    "id": sid,
                    "field": field,
                    "issue": "duplicate_refs",
                    "values": vals,
                })

            for j, v in enumerate(vals[1:], start=1):
                sv = as_text(v)
                if not is_valid_option(sv):
                    issues.append({
                        "index": i,
                        "id": sid,
                        "field": field,
                        "issue": "invalid_format",
                        "value": sv,
                        "position": j,
                    })
                if same_referent(gtv, sv):
                    issues.append({
                        "index": i,
                        "id": sid,
                        "field": field,
                        "issue": "distractor_same_ref_as_gt",
                        "value": sv,
                        "ground_truth": gtv,
                        "position": j,
                    })
    return issues


def main():
    stage2 = json.loads(STAGE2_PATH.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(stage2, list):
        raise ValueError("Stage2 JSON must be list")
    if not isinstance(ref, list):
        raise ValueError("Reference JSON must be list")

    ref_entries, scenario_pool = build_reference_index(ref)

    final_items = []
    traces = []
    for rec in stage2:
        item, trace = build_record(rec, ref_entries, scenario_pool)
        final_items.append(item)
        traces.append(trace)

    issues = validate(final_items)

    FINAL_PATH.write_text(json.dumps(final_items, ensure_ascii=False, indent=2), encoding="utf-8")
    CHECK_REPORT_PATH.write_text(
        json.dumps(
            {"total": len(final_items), "issue_count": len(issues), "issues": issues},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # summarize distractor sources
    source_counts = defaultdict(int)
    for t in traces:
        for x in t.get("subject_trace", []):
            source_counts[x["source"].split(":")[0]] += 1
        for x in t.get("target_trace", []):
            source_counts[x["source"].split(":")[0]] += 1

    DISTRACTOR_REPORT_PATH.write_text(
        json.dumps(
            {
                "total_records": len(traces),
                "source_counts": dict(sorted(source_counts.items(), key=lambda kv: kv[0])),
                "traces": traces,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {FINAL_PATH}")
    print(f"Saved: {CHECK_REPORT_PATH}")
    print(f"Saved: {DISTRACTOR_REPORT_PATH}")
    print(f"Issue count: {len(issues)}")
    print(f"Source counts: {dict(source_counts)}")


if __name__ == "__main__":
    main()
