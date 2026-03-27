import json
import re
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(r"export\review_assign_all_without_abandoned_unique_replaced_with_high_level_sample.json")
OUTPUT_PATH = Path(r"D:\NUS\ACMm\Data-annotation\export\All_exported_labels_new_format.json")
VALID_SCENARIOS = {"affection", "attitude", "intent"}
PLACEHOLDER_MARKERS = ("use non", "distractor", "placeholder", "tbd")
LIGHT_STOPWORDS = {"the", "a", "an"}
REFERENT_NOISE_TOKENS = {"person", "someone", "somebody", "individual", "entity"}
FALLBACK_CANDIDATES = {
    "subject": [
        "the bystander",
        "the coworker",
        "the manager",
        "the customer",
        "the friend",
        "the teammate",
        "the observer",
        "the passerby",
    ],
    "target": [
        "the conversation topic",
        "the current situation",
        "the previous remark",
        "the store policy",
        "the group discussion",
        "the task at hand",
        "the background activity",
        "the nearby object",
    ],
}


def as_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_nullish(value) -> bool:
    text = as_text(value)
    return not text or text.upper() == "NULL"


def first_non_nullish(*values) -> str:
    for value in values:
        if not is_nullish(value):
            return as_text(value)
    return ""


def is_placeholder_option(value: str) -> bool:
    text = as_text(value).lower()
    return any(marker in text for marker in PLACEHOLDER_MARKERS)


def normalize_referent(value: str) -> str:
    text = as_text(value).lower()
    if not text:
        return "__empty__"
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok]
    filtered = [
        tok
        for tok in tokens
        if tok not in LIGHT_STOPWORDS and tok not in REFERENT_NOISE_TOKENS
    ]
    if not filtered:
        return "__empty__"
    return " ".join(filtered)


def pick_scenario(inp: dict, out: dict) -> str:
    candidates = [out.get("scenario"), out.get("situation"), inp.get("scenario")]
    for value in candidates:
        text = as_text(value).lower()
        if text in VALID_SCENARIOS:
            return text
    return as_text(candidates[0]).lower()


def pick_media(inp: dict) -> dict:
    media_path = as_text(inp.get("media_path"))
    media_path_local = as_text(inp.get("media_path_local"))
    audio_path = as_text(inp.get("audio_path"))
    audio_path_local = as_text(inp.get("audio_path_local"))
    url = as_text(inp.get("url"))
    path = as_text(inp.get("path"))

    has_video_signal = any(
        not is_nullish(value)
        for value in (media_path, media_path_local, audio_path, audio_path_local)
    )
    if not has_video_signal and (url.lower().endswith(".mp4") or path.lower().endswith(".mp4")):
        has_video_signal = True

    if has_video_signal:
        video_url = first_non_nullish(media_path, url if url.lower().endswith(".mp4") else "")
        video_path = first_non_nullish(media_path_local, path if path.lower().endswith(".mp4") else "")
        return {
            "video_url": video_url,
            "audio_url": first_non_nullish(audio_path),
            "audio_caption": as_text(inp.get("audio_caption")),
            "video_path": video_path,
            "audio_path": first_non_nullish(audio_path_local),
        }

    return {
        "image_url": first_non_nullish(url, media_path),
        "image_path": first_non_nullish(path, media_path_local),
    }


def pick_mechanism_or_label(out: dict, scenario: str, key: str) -> str:
    primary = as_text(out.get(key))
    if primary and primary.upper() != "NULL":
        return primary
    scenario_key = f"{key}_{scenario.capitalize()}"
    fallback = as_text(out.get(scenario_key))
    if fallback and fallback.upper() != "NULL":
        return fallback
    for suffix in ("Affection", "Attitude", "Intent"):
        cross_key = f"{key}_{suffix}"
        cross_value = as_text(out.get(cross_key))
        if cross_value and cross_value.upper() != "NULL":
            return cross_value
    return ""


def context_candidates(item: dict, field: str) -> list[str]:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    text = as_text(input_obj.get("text"))
    lower_text = text.lower()
    media_obj = input_obj.get("media", {}) if isinstance(input_obj.get("media"), dict) else {}
    media_hint = " ".join(
        [
            as_text(media_obj.get("video_url")),
            as_text(media_obj.get("image_url")),
        ]
    ).lower()

    cands: list[str] = []
    if field == "subject":
        if " you " in f" {lower_text} ":
            cands.append("the addressee")
        if " i " in f" {lower_text} " or lower_text.startswith("i "):
            cands.append("the speaker")
        if any(tag in media_hint for tag in ("store", "shop", "superstore")):
            cands.extend(["the store employee", "the customer"])
    else:
        if " you " in f" {lower_text} ":
            cands.append("the addressee")
        if any(word in lower_text for word in ("policy", "rule", "duty")):
            cands.append("the policy")
        if any(tag in media_hint for tag in ("store", "shop", "superstore")):
            cands.extend(["the checkout process", "the store policy"])
        cands.extend(["the conversation topic", "the current situation"])

    names = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    for name in names[:3]:
        if field == "subject":
            cands.append(name)
        else:
            cands.append(f"the person named {name}")
    return cands


def convert_one(sample: dict) -> dict:
    inp = sample.get("input", {})
    out = sample.get("output", {})

    scenario = pick_scenario(inp, out)
    sample_id = first_non_nullish(inp.get("id"), inp.get("samples_id"), sample.get("id"))

    subjects = [
        as_text(out.get("subject")),
        as_text(out.get("subject1")),
        as_text(out.get("subject2")),
        as_text(out.get("subject3")),
    ]
    targets = [
        as_text(out.get("target")),
        as_text(out.get("target1")),
        as_text(out.get("target2")),
        as_text(out.get("target3")),
    ]

    return {
        "id": sample_id,
        "input": {
            "scenario": scenario,
            "text": as_text(inp.get("text")),
            "media": pick_media(inp),
        },
        "options": {
            "subject": subjects,
            "target": targets,
        },
        "ground_truth": {
            "subject": as_text(out.get("subject")),
            "target": as_text(out.get("target")),
            "mechanism": pick_mechanism_or_label(out, scenario, "mechanism"),
            "label": pick_mechanism_or_label(out, scenario, "label"),
        },
        "diversity": {
            "domain": as_text(out.get("domain")),
            "culture": as_text(out.get("culture")),
        },
    }


def collect_option_issues(sample_id: str, field: str, values: list[str], ground_truth: str) -> list[str]:
    issues: list[str] = []
    if len(values) != 4:
        issues.append(
            f"Sample {sample_id}: {field} options length is {len(values)} (expected 4)."
        )
    if len([x for x in values if x]) != 4:
        issues.append(
            f"Sample {sample_id}: {field} options are not 4 complete entries."
        )

    keys = [normalize_referent(x) for x in values]
    if len(set(keys)) != 4:
        issues.append(
            f"Sample {sample_id}: {field} options contain duplicate referents."
        )

    gt_key = normalize_referent(ground_truth)
    if gt_key != "__empty__":
        gt_hits = sum(1 for k in keys if k == gt_key)
        if gt_hits != 1:
            issues.append(
                f"Sample {sample_id}: {field} should contain exactly one option aligned with ground truth referent."
            )
    return issues


def build_candidate_pools(converted: list[dict]) -> dict[tuple[str, str], list[str]]:
    pools: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen_keys: dict[tuple[str, str], set[str]] = defaultdict(set)

    for item in converted:
        input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
        scenario = as_text(input_obj.get("scenario")).lower()
        options = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
        for field in ("subject", "target"):
            values = options.get(field, [])
            if not isinstance(values, list):
                continue
            for value in values:
                text = as_text(value)
                if not text or is_placeholder_option(text):
                    continue
                key = normalize_referent(text)
                if key == "__empty__":
                    continue
                for bucket in ((scenario, field), ("all", field)):
                    if key in seen_keys[bucket]:
                        continue
                    seen_keys[bucket].add(key)
                    pools[bucket].append(text)
    return pools


def pick_replacement(
    item: dict,
    field: str,
    used_keys: set[str],
    gt_key: str,
    pools: dict[tuple[str, str], list[str]],
) -> tuple[str, str]:
    input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    scenario = as_text(input_obj.get("scenario")).lower()

    candidates: list[str] = []
    candidates.extend(context_candidates(item, field))
    candidates.extend(pools.get((scenario, field), []))
    candidates.extend(pools.get(("all", field), []))
    candidates.extend(FALLBACK_CANDIDATES[field])

    for candidate in candidates:
        text = as_text(candidate)
        if not text or is_placeholder_option(text):
            continue
        key = normalize_referent(text)
        if key == "__empty__":
            continue
        if key == gt_key or key in used_keys:
            continue
        return text, key

    base = "alternative subject" if field == "subject" else "alternative target"
    idx = 1
    while True:
        candidate = f"{base} {idx}"
        key = normalize_referent(candidate)
        if key != gt_key and key not in used_keys:
            return candidate, key
        idx += 1


def repair_field_options(
    item: dict,
    field: str,
    pools: dict[tuple[str, str], list[str]],
) -> tuple[list[str], list[dict]]:
    options_obj = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
    raw_options = options_obj.get(field, [])
    options = [as_text(x) for x in raw_options] if isinstance(raw_options, list) else []
    options = options[:4]
    while len(options) < 4:
        options.append("")

    gt_obj = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}
    ground_truth = as_text(gt_obj.get(field))
    gt_key = normalize_referent(ground_truth)

    keys = [normalize_referent(x) for x in options]
    gt_keeper = None
    if gt_key != "__empty__":
        for i, value in enumerate(options):
            if normalize_referent(value) == gt_key and as_text(value).lower() == ground_truth.lower():
                gt_keeper = i
                break
        if gt_keeper is None:
            for i, key in enumerate(keys):
                if key == gt_key:
                    gt_keeper = i
                    break
    if gt_keeper is None:
        gt_keeper = 0
        if ground_truth:
            options[0] = ground_truth
    if ground_truth:
        options[gt_keeper] = ground_truth
    keys = [normalize_referent(x) for x in options]

    replace_reasons: dict[int, set[str]] = defaultdict(set)
    for i, value in enumerate(options):
        if not value:
            replace_reasons[i].add("empty_option")
        if is_placeholder_option(value):
            replace_reasons[i].add("placeholder_option")

    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        key_to_indices[key].append(i)

    for key, indices in key_to_indices.items():
        if key == "__empty__":
            for i in indices:
                replace_reasons[i].add("empty_referent")
            continue
        if gt_key != "__empty__" and key == gt_key:
            for i in indices:
                if i != gt_keeper:
                    replace_reasons[i].add("same_as_ground_truth")
            continue
        for i in indices[1:]:
            replace_reasons[i].add("duplicate_referent")

    replace_reasons.pop(gt_keeper, None)

    used_keys: set[str] = set()
    for i, value in enumerate(options):
        if i in replace_reasons:
            continue
        key = normalize_referent(value)
        if key != "__empty__":
            used_keys.add(key)
    if gt_key != "__empty__":
        used_keys.add(gt_key)

    changes: list[dict] = []
    for i in sorted(replace_reasons.keys()):
        old_value = options[i]
        new_value, new_key = pick_replacement(item, field, used_keys, gt_key, pools)
        options[i] = new_value
        used_keys.add(new_key)
        changes.append(
            {
                "option_index": i,
                "old_value": old_value,
                "new_value": new_value,
                "reasons": sorted(replace_reasons[i]),
            }
        )

    return options, changes


def repair_options(converted: list[dict]) -> list[dict]:
    pools = build_candidate_pools(converted)
    repair_records: list[dict] = []

    for idx, item in enumerate(converted):
        item_options = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
        sample_changes: dict[str, list[dict]] = {}

        for field in ("subject", "target"):
            before = list(item_options.get(field, [])) if isinstance(item_options.get(field, []), list) else []
            fixed, changes = repair_field_options(item, field, pools)
            item_options[field] = fixed
            if changes:
                sample_changes[field] = changes
                sample_changes[f"{field}_before"] = before
                sample_changes[f"{field}_after"] = fixed

        item["options"] = item_options

        if sample_changes:
            input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
            media_obj = input_obj.get("media", {}) if isinstance(input_obj.get("media"), dict) else {}
            repair_records.append(
                {
                    "index": idx,
                    "id": item.get("id", ""),
                    "scenario": input_obj.get("scenario", ""),
                    "text": input_obj.get("text", ""),
                    "video_url": media_obj.get("video_url", ""),
                    "image_url": media_obj.get("image_url", ""),
                    "changes": sample_changes,
                }
            )

    return repair_records


def build_issue_records(converted: list[dict]) -> list[dict]:
    issue_records: list[dict] = []
    for idx, item in enumerate(converted):
        options = item.get("options", {}) if isinstance(item.get("options"), dict) else {}
        gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}
        sample_id = as_text(item.get("id"))
        subject_values = options.get("subject", []) if isinstance(options.get("subject"), list) else []
        target_values = options.get("target", []) if isinstance(options.get("target"), list) else []
        subject_issues = collect_option_issues(
            sample_id,
            "subject",
            [as_text(x) for x in subject_values],
            as_text(gt.get("subject")),
        )
        target_issues = collect_option_issues(
            sample_id,
            "target",
            [as_text(x) for x in target_values],
            as_text(gt.get("target")),
        )
        issues = subject_issues + target_issues
        if not issues:
            continue
        input_obj = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
        media_obj = input_obj.get("media", {}) if isinstance(input_obj.get("media"), dict) else {}
        issue_records.append(
            {
                "index": idx,
                "id": sample_id,
                "issues": issues,
                "options": options,
                "ground_truth": gt,
                "input_text": input_obj.get("text", ""),
                "video_url": media_obj.get("video_url", ""),
                "image_url": media_obj.get("image_url", ""),
            }
        )
    return issue_records


def main() -> None:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {INPUT_PATH}, got {type(raw).__name__}.")

    converted = [convert_one(sample) for sample in raw]
    repair_records = repair_options(converted)
    issue_records = build_issue_records(converted)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

    repair_report_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_dedup_repair_report.json")
    repair_report_path.write_text(
        json.dumps(repair_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    issue_report_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_manual_fix_report.json")
    issue_report_path.write_text(
        json.dumps(issue_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Converted {len(converted)} samples.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Dedup-repair records: {len(repair_records)}")
    print(f"Dedup-repair report: {repair_report_path}")
    print(f"Manual-fix issue records: {len(issue_records)}")
    print(f"Manual-fix report: {issue_report_path}")


if __name__ == "__main__":
    main()
