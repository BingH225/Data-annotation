import json
import re
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(r"D:\NUS\ACMm\Data-annotation\Task\05_adding_samples")
INPUT_PATH = BASE_DIR / "Zuo_video_primary_day03_export_stage2.json"
OUTPUT_PATH = BASE_DIR / "Zuo_video_primary_day03_export_manual40.json"


MANUAL_OPTIONS: Dict[str, Dict[str, List[str]]] = {
    "affection_1444": {
        "subject": ["woman in white", "left blond friend", "party guest"],
        "target": ["the bridal tension", "the room mood", "the party banter"],
    },
    "affection_0320": {
        "subject": ["bald coworker", "standing supervisor", "nearby cashier"],
        "target": ["the manager on duty", "the checkout line", "the customer complaint"],
    },
    "affection_1056": {
        "subject": ["older supervisor", "office visitor", "desk colleague"],
        "target": ["the office schedule", "the paperwork backlog", "the budget review"],
    },
    "attitude_1510": {
        "subject": ["seated friend", "woman on couch", "kissing woman"],
        "target": ["the kissing scene", "the sister", "the room audience"],
    },
    "affection_0568": {
        "subject": ["male coworker", "female customer", "security guard"],
        "target": ["the work schedule", "the checkout queue", "the store policy"],
    },
    "affection_1493": {
        "subject": ["blonde friend", "man in vest", "nearby waitress"],
        "target": ["the blonde friend", "the slot machines", "the night out"],
    },
    "affection_1379": {
        "subject": ["suited listener", "hallway staffer", "doorway observer"],
        "target": ["the schedule question", "the sales numbers", "the hallway meeting"],
    },
    "attitude_1634": {
        "subject": ["woman at doorway", "man by kitchen", "nearby roommate"],
        "target": ["the apartment doorway", "his awkward confession", "the kitchen mess"],
    },
    "affection_0066": {
        "subject": ["female coworker", "security guard", "checkout customer"],
        "target": ["the security guard", "the front counter", "the customer line"],
    },
    "affection_1069": {
        "subject": ["female interviewer", "office manager", "front desk clerk"],
        "target": ["the hiring manager", "the office politics", "the resume review"],
    },
    "affection_1088": {
        "subject": ["woman by display", "passing shopper", "seasonal coworker"],
        "target": ["the stockroom mess", "the customer traffic", "the return counter"],
    },
    "affection_1313": {
        "subject": ["woman in purple", "man on floor", "kitchen observer"],
        "target": ["the kitchen stunt", "the wrapped present", "the dinner plans"],
    },
    "affection_0131": {
        "subject": ["bald coworker", "male customer", "cash wrap clerk"],
        "target": ["his current date", "the breakup story", "the apology text"],
    },
    "affection_1205": {
        "subject": ["man with glasses", "restaurant patron", "table waitress"],
        "target": ["the party room", "the tequila bottle", "the dinner crowd"],
    },
    "affection_0014": {
        "subject": ["Amy Sosa", "seated officer", "bald coworker"],
        "target": ["the training session", "the holiday display", "the apology demand"],
    },
    "affection_0005": {
        "subject": ["female coworker", "clinic nurse", "store customer"],
        "target": ["the ice pack", "the yam incident", "the break room wall"],
    },
    "affection_0858": {
        "subject": ["man with clipboard", "banquet guest", "event planner"],
        "target": ["the paperwork", "the party table", "the catering line"],
    },
    "affection_0187": {
        "subject": ["girl in white", "girl in black", "man in suit"],
        "target": ["the purple doll", "the school dance", "the prize table"],
    },
    "affection_0567": {
        "subject": ["pink vest coworker", "bald manager", "nearby shopper"],
        "target": ["the staff meeting", "the shift assignment", "the checkout issue"],
    },
    "affection_0865": {
        "subject": ["female coworker", "floor manager", "counter customer"],
        "target": ["the workload", "the staff room", "the open shift"],
    },
    "affection_1514": {
        "subject": ["woman in red", "apartment roommate", "doorway guest"],
        "target": ["the notebook joke", "the living room mood", "the evening plans"],
    },
    "affection_0903": {
        "subject": ["male coworker", "pink blouse shopper", "nearby customer"],
        "target": ["the broken display", "the checkout delay", "the crowd gathering"],
    },
    "affection_0953": {
        "subject": ["seated coworker", "shift manager", "stockroom worker"],
        "target": ["the meeting topic", "the new policy", "the staffing issue"],
    },
    "affection_1331": {
        "subject": ["woman in tan", "woman in black", "man on couch"],
        "target": ["the balloon joke", "the cartoon debate", "the table discussion"],
    },
    "affection_0916": {
        "subject": ["blue vest trainee", "yellow vest trainee", "male floor worker"],
        "target": ["the apology request", "the paperwork stack", "the office rules"],
    },
    "affection_0706": {
        "subject": ["male employee", "seated patron", "front desk clerk"],
        "target": ["the phone call", "the strange behavior", "the party rumor"],
    },
    "affection_0163": {
        "subject": ["man across desk", "office assistant", "department chief"],
        "target": ["the paperwork", "the office policy", "the meeting request"],
    },
    "affection_1173": {
        "subject": ["red haired woman", "movie usher", "man in aisle"],
        "target": ["the lost earring", "the theater date", "the empty seat"],
    },
    "affection_1325": {
        "subject": ["woman beside him", "roommate friend", "older parent"],
        "target": ["the pillow joke", "the parent story", "the couch conversation"],
    },
    "affection_1384": {
        "subject": ["standing roommate", "woman at doorway", "apartment guest"],
        "target": ["the made up game", "the apartment scene", "the beer bottle"],
    },
    "affection_1216": {
        "subject": ["seated woman", "woman in green", "roommate guest"],
        "target": ["the coffee pot", "the color advice", "the kitchen table"],
    },
    "affection_1453": {
        "subject": ["Ross on bed", "apartment roommate", "bedroom friend"],
        "target": ["the breakup news", "the phone call", "the other relationship"],
    },
    "affection_1199": {
        "subject": ["Sheldon", "dinner guest", "roommate listener"],
        "target": ["the restaurant choice", "the dinner invitation", "the takeout joke"],
    },
    "affection_0072": {
        "subject": ["new employee", "office manager", "front desk clerk"],
        "target": ["the meeting notes", "the training packet", "the filing deadline"],
    },
    "affection_1515": {
        "subject": ["woman by fridge", "apartment roommate", "doorway guest"],
        "target": ["the apartment joke", "the surprise visit", "the afternoon plans"],
    },
    "affection_0218": {
        "subject": ["female coworker", "male customer", "nearby cashier"],
        "target": ["the shopping cart", "the store promotion", "the work shift"],
    },
    "affection_0311": {
        "subject": ["man with booklet", "kitchen roommate", "nearby friend"],
        "target": ["the open cookbook", "the kitchen mess", "the roommate advice"],
    },
    "affection_0172": {
        "subject": ["female coworker", "older customer", "store clerk"],
        "target": ["the head injury", "the apology", "the break room chat"],
    },
    "affection_1360": {
        "subject": ["Chandler", "apartment guest", "couch friend"],
        "target": ["the surname joke", "the apartment story", "the hand gesture"],
    },
    "affection_1147": {
        "subject": ["bearded coworker", "woman in glasses", "training supervisor"],
        "target": ["the training session", "the staff meeting", "the seating arrangement"],
    },
}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def pick_scenario(record: Dict[str, Any]) -> str:
    out = record.get("output", {}) if isinstance(record.get("output"), dict) else {}
    for key in ("situation", "scenario"):
        value = as_text(out.get(key)).lower()
        if value:
            return value
    return ""


def pick_mechanism(out: Dict[str, Any], scenario: str) -> str:
    primary = as_text(out.get("mechanism"))
    if primary and primary.upper() != "NULL":
        return primary
    specific = as_text(out.get(f"mechanism_{scenario.capitalize()}"))
    if specific and specific.upper() != "NULL":
        return specific
    return ""


def pick_label(out: Dict[str, Any], scenario: str) -> str:
    primary = as_text(out.get("label"))
    if primary and primary.upper() != "NULL":
        return primary
    specific = as_text(out.get(f"label_{scenario.capitalize()}"))
    if specific and specific.upper() != "NULL":
        return specific
    return ""


def normalize_ref(value: str) -> str:
    text = as_text(value).lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return " ".join(tokens)


def validate_options(sample_id: str, ground_truth: str, distractors: List[str]) -> None:
    if len(distractors) != 3:
        raise ValueError(f"{sample_id}: expected 3 distractors, got {len(distractors)}")
    gt_key = normalize_ref(ground_truth)
    seen = set()
    for option in distractors:
        if any(ch in option for ch in ("/", "\\", "(", ")")):
            raise ValueError(f"{sample_id}: invalid punctuation in option {option!r}")
        if len(option.split()) > 5:
            raise ValueError(f"{sample_id}: option too long {option!r}")
        key = normalize_ref(option)
        if key == gt_key:
            raise ValueError(f"{sample_id}: distractor matches ground truth referent {option!r}")
        if key in seen:
            raise ValueError(f"{sample_id}: duplicate distractor {option!r}")
        seen.add(key)


def main() -> None:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list.")

    selected = raw[:40]
    converted: List[Dict[str, Any]] = []

    for record in selected:
        inp = record.get("input", {}) if isinstance(record.get("input"), dict) else {}
        out = record.get("output", {}) if isinstance(record.get("output"), dict) else {}
        sample_id = as_text(inp.get("id") or inp.get("samples_id"))
        manual = MANUAL_OPTIONS.get(sample_id)
        if not manual:
            raise KeyError(f"Missing manual options for {sample_id}")

        gt_subject = as_text(out.get("subject"))
        gt_target = as_text(out.get("target"))
        validate_options(sample_id, gt_subject, manual["subject"])
        validate_options(sample_id, gt_target, manual["target"])

        scenario = pick_scenario(record)
        converted.append(
            {
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
                    "subject": [gt_subject] + manual["subject"],
                    "target": [gt_target] + manual["target"],
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
        )

    OUTPUT_PATH.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(converted)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
