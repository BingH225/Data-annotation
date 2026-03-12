import json
from pathlib import Path


INPUT_PATH = Path(r"D:\NUS\ACMm\Data-annotation\export\exported_labels.json")
OUTPUT_PATH = Path(r"D:\NUS\ACMm\Data-annotation\export\exported_labels_new_format.json")
VALID_SCENARIOS = {"affection", "attitude", "intent"}


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


def convert_one(sample: dict) -> dict:
    inp = sample.get("input", {})
    out = sample.get("output", {})

    scenario = pick_scenario(inp, out)
    sample_id = first_non_nullish(inp.get("id"), inp.get("samples_id"))

    subjects = [as_text(out.get("subject")), as_text(out.get("subject1")), as_text(out.get("subject2")), as_text(out.get("subject3"))]
    targets = [as_text(out.get("target")), as_text(out.get("target1")), as_text(out.get("target2")), as_text(out.get("target3"))]

    if len([x for x in subjects if x]) != 4:
        raise ValueError(f"Sample {sample_id}: subject options are not 4 complete entries.")
    if len([x for x in targets if x]) != 4:
        raise ValueError(f"Sample {sample_id}: target options are not 4 complete entries.")

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
    }


def main() -> None:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {INPUT_PATH}, got {type(raw).__name__}.")

    converted = [convert_one(sample) for sample in raw]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Converted {len(converted)} samples.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
