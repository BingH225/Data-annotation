import os
import json
import time
import random
import base64
import hashlib
import subprocess
import re
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL_NAME = "gpt-5-mini"

IMAGE_JSON_PATH = "/hpctmp/e1561245/top3snt/top_3_subntar_image_gpt5mini/prelabel_slim_results_newhf_intmp2.json"
VIDEO_JSON_PATH = "/hpctmp/e1561245/top3snt/top_3_subntar_video_gpt5mini/prelabel_slim_results_newhf_intmp2.json"

SAMPLE_SIZE = 100
# 42
RANDOM_SEED = 4124
MAX_WORKERS = 8
MAX_RETRIES = 3

SITUATIONS = ["affection", "attitude", "intent"]
DOMAINS = ["Online & Social Media", "Public & Service", "Workplace", "Intimate Relationships", "Family Conversations",
           "Friend Group", "Education & Campus", "Friendship Interactions"]
CULTURES = ["General Culture", "Arab Culture", "American Culture", "Muslim Culture", "African American Culture",
            "Jewish Culture", "Indian Culture", "East Asian Culture"]

# ==========================================
# 动态创建输出文件夹
# ==========================================
OUTPUT_DIR = Path(f"./{MODEL_NAME}_{SAMPLE_SIZE}_With_Situation_DnC_6_seed{RANDOM_SEED}_14:15")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DETAILED_FILE = OUTPUT_DIR / "evaluation_predictions_detailed.json"
OUTPUT_METRICS_FILE = OUTPUT_DIR / "evaluation_metrics_report.json"
OUTPUT_FAILED_FILE = OUTPUT_DIR / "evaluation_failures.json"
OUTPUT_PLOT_OVERALL_FILE = OUTPUT_DIR / "evaluation_metrics_overall_bar.png"
OUTPUT_PLOT_SITUATION_FILE = OUTPUT_DIR / "evaluation_metrics_by_situation_bar.png"
OUTPUT_PLOT_DOMAIN_FILE = OUTPUT_DIR / "evaluation_metrics_by_domain_bar.png"
OUTPUT_PLOT_CULTURE_FILE = OUTPUT_DIR / "evaluation_metrics_by_culture_bar.png"
OUTPUT_TABLE_FILE = OUTPUT_DIR / "evaluation_metrics_summary_table.png"
OUTPUT_STRICT_TABLE_FILE = OUTPUT_DIR / "evaluation_metrics_strict_table.png"

# 全局有效列表，用于错误统计校验
VALID_MECHANISMS = {
    "affection": ["multimodal_incongruity", "figurative_semantics", "affective_deception", "socio_cultural_dependency"],
    "intent": ["prosocial_deception", "malicious_manipulation", "expressive_aggression", "benevolent_provocation"],
    "attitude": ["dominant_affiliation", "dominant_detachment", "protective_distancing", "submissive_alignment"]
}
VALID_LABELS = {
    "affection": ["happy", "sad", "disgusted", "angry", "fearful", "bad"],
    "attitude": ["supportive", "appreciative", "sympathetic", "neutral",
                 "indifferent", "concerned", "skeptical", "dismissive", "disapproving", "contemptuous", "hostile"],
     # "intent": ["mitigate", "manipulate", "ridicule", "condemn", "alienate"]
    "intent": ["mitigate", "intimidate", "alienate", "mock", "denounce", "troll", "dominate", "condemn"]
}

# ==========================================
# System Prompt (统一版本：根据输入提供的 situation/domain/culture 做判断)
# ==========================================
SYSTEM_PROMPT = """
Your task is to objectively classify the implicit social dynamics in the provided text-image/video pair based on a GIVEN "situation", "domain", and "culture". This is a strict technical annotation task for sociological research.

NOTE: The terminology used in the categories below (e.g., "malicious_manipulation", "deception", "aggression") refers strictly to abstract theoretical constructs in pragmatics. They are academic codes for communicative strategies, NOT moral judgments or literal accusations against any individuals depicted.

# Academic Coding Framework

[The "situation", "domain", and "culture" are provided in the user prompt. Do not predict them.]

CRITICAL RULE: You MUST strictly adhere to the decision mapping below. You are STRICTLY FORBIDDEN from selecting a mechanism or label that belongs to a different situation. DO NOT confuse mechanisms with labels.

1. Entity Role Definition (decide these first):
- "subject": the implicit initiator/holder of the social act in this sample (the one expressing the underlying affect, intent, or stance).
- "target": the implicit receiver/object toward which that affect, intent, or stance is directed.
- You MUST output subject/target as slot IDs only: subject in ["subject0", "subject1", "subject2", "subject3"], target in ["target0", "target1", "target2", "target3"].
- Do NOT output descriptive text for subject/target.

2. "mechanism": Based on the provided "situation", choose exactly ONE theoretical code from the specific section below:

=== IF SITUATION IS "affection" ===
(DO NOT use attitude or intent mechanisms. ONLY choose from these 4):
- "multimodal_incongruity": Implicit affection arises from polarity conflict or mutual exclusion between modalities (text vs image, or image-context vs text), such that the literal meaning is negated or reframed by the other modality. Key signature: "What is said" and "what is shown" cannot both be true in the same frame -> the affective state is inferred from the conflict.
- "figurative_semantics": The affective state is conveyed via source->target conceptual mapping rather than direct emotion words or standard displays. Key signature: The sample "talks about X" but means an affective state through metaphor, symbol, hyperbole, understatement, or poetic imagery.
- "affective_deception": The affective state is deliberately masked (performed neutrality/harshness), but involuntary cues "leak" the underlying affect. Key signature: "Displayed affect" != "true affective state" inferred from leakage.
- "socio_cultural_dependency": The affective state can only be interpreted correctly using external world knowledge (memes, events, cultural codes, relationship norms). Key signature: The pair is semantically opaque without a shared cultural reference that encodes affect indirectly.

=== IF SITUATION IS "intent" ===
(DO NOT use affection or attitude mechanisms. ONLY choose from these 4):
- "prosocial_deception": A benevolent concealment where the speaker prioritizes social harmony/face-saving over factual accuracy, masking friction with surface agreement or positivity. Key signature: Surface message appears supportive/neutral, but context implies the true goal is to protect feelings, avoid conflict, or maintain relationship stability.
- "malicious_manipulation": A zero-sum strategic intent where the speaker exploits human vulnerabilities (vanity, pity, guilt) while packaging harm/control as help, care, or morality.
- "expressive_aggression": Indirect aggression used to vent dissatisfaction or establish dominance when direct confrontation is risky; the target is displaced or blurred.
- "benevolent_provocation": A tool-like disguise based on information asymmetry; the speaker misleads or provokes to trigger a desired action or reveal truthful capability.

=== IF SITUATION IS "attitude" ===
(DO NOT use affection or intent mechanisms. ONLY choose from these 4):
- "dominant_affiliation": Surface friendliness/acceptance is used to assert superiority; "closeness" is granted from above. Key signature: warmth-like form + downward positioning of the target.
- "dominant_detachment": High-ground stance + emotional cutoff; negation, exclusion, contempt, or moral/intellectual judgment. Key signature: dismissal/negation framed as reasonableness, superiority, or moral clarity.
- "protective_distancing": Lower-power self-protection via non-commitment, disengagement, or guarded skepticism. Key signature: surface politeness/open-ness while retracting commitment, emotion, or shared premises.
- "submissive_alignment": Self-lowering to secure acceptance/safety; over-agreement or over-yielding. Key signature: deference, self-devaluation, or surrendering one's stance.


3. "label": Based on the provided "situation", choose EXACTLY ONE from the specific section below:

=== IF SITUATION IS "affection" ===
(Choose ONE of the following true emotions):
- "Happy": A positive, pleasant affective state including satisfaction, joy, interest, relaxation, or confidence; typically includes a positive appraisal of the current situation (e.g., feeling accepted/recognized, hopeful/motivated).
- "Sad": A negative low mood linked to loss, helplessness, sorrow, loneliness, hurt, or guilt; typically includes negative appraisal of loss/failure/relationship setback, possibly withdrawal/crying/low motivation.
- "Disgusted": Strong aversion/rejection centered on disgust/repulsion (physiological like smell/food, or social-moral like hypocrisy/offense); tends to "want to avoid/reject/deny".
- "Angry": Strong unpleasant, confrontational tendency triggered by offense, obstruction, or unfairness; includes blame, hostility, irritation, rage, frustration; often paired with control/retaliation/argument orientation.
- "Fearful": Negative affect centered on threat/insecurity; includes fear, tension, worry, panic, vigilance; corresponds to expectation of danger/being hurt/"something will go wrong"; tends toward avoidance, seeking protection, or heightened alertness.
- "Bad": A broad negative state used when the affect is negative but does not clearly fit Sad/Angry/Disgusted/Fearful; can include discomfort, fatigue, stress, boredom, coldness, numbness, helplessness, confusion, or "feeling off".

=== IF SITUATION IS "attitude" ===
(Choose ONE of the following stances toward the target):
- "Supportive": explicitly defends the subject or affirms their legitimacy/value; takes the subject's side.
- "Appreciative": gives positive evaluation of the subject's abilities/qualities/achievements; not necessarily taking sides.
- "Sympathetic": shows understanding/empathy for the subject's unfavorable situation; downplays their responsibility.
- "Neutral": no clear stance positioning.
- "Indifferent": disengaged but not necessarily skeptical/negative.
- "Concerned": caution/care about outcome, not necessarily warm.
- "Skeptical": doubts assumptions/claims; guarded belief.
- "Dismissive": downplays the target's point/feelings; "not worth it."
- "Disapproving": negative evaluation of behavior/choice without contempt.
- "Contemptuous": demeaning/derogatory high-ground; target is beneath respect.
- "Hostile": aggressive antagonism/threatening tone (only if grounded).

=== IF SITUATION IS "intent" ===
(Choose ONE of the following true communicative goals):
- "Mitigate": Intent to de-escalate and save face. The subject actively tries to reduce hostility, offer a compromise, show empathy, or smooth over an awkward situation to maintain a cooperative relationship.
- "Intimidate": Intent to coerce through fear or consequences. The subject implicitly or explicitly threatens the target with future harm, retaliation, or negative outcomes to force them to back down or comply.
- "Alienate": Intent to isolate and dehumanize. The subject strips the target of equal status by reducing them to a negative group stereotype or labeling them as an unacceptable "outsider" (us vs. them).
- "Mock": Intent to entertain at the target's expense. The subject treats the target's flaws or awkwardness as a joke. The primary goal is amusement and making the target look foolish, rather than expressing deep hatred.
- "Denounce": Intent to socially punish via exposure. The subject brings the target's actions to public attention, rallying an audience or social pressure to shame, cancel, or socially isolate the target.
- "Troll": Intent to attack or provoke using digital culture as a weapon. The subject uses internet memes, heavy irony, or absurd humor to disguise their hostility, aiming to anger the target while maintaining "plausible deniability".
- "Dominate": Intent to establish power or intellectual asymmetry. The subject uses condescension, pulls rank, or acts patronizingly to show superiority ("I am above you"), forcing the target into a subordinate position.
- "Condemn": Intent to judge based on morals or rules. The subject takes the moral or logical high ground to declare the target's actions as wrong, unethical, or invalid. It is about playing the "righteous judge".

# Output Format Requirements
You may reason internally before deciding, but DO NOT output any reasoning.
Make one integrated decision for all four fields and output the final answer once.
Output ONLY a strict JSON object with EXACTLY 4 keys. You MUST adhere to the exact values requested. Use this exact schema:
{
  "mechanism": "<insert ONE valid mechanism here>",
  "label": "<insert ONE valid label here>",
  "subject": "<MUST be subject0, subject1, subject2, or subject3>",
  "target": "<MUST be target0, target1, target2, or target3>"
}
Do NOT write markdown blocks (no ```json).
"""

# - "Mitigate": To reduce the intensity of conflict or interpersonal threat by de-escalating tension, preserving face, and avoiding confrontation through conciliatory or protective communicative moves.
# - "Manipulate": To influence or control another person's behavior or choices through pressure, coercion, or dominance-oriented tactics that constrain the other party's autonomy.
# - "Ridicule": To mock or deride a target, often in an entertaining or humorous frame, in order to make them appear foolish, inferior, or socially discreditable.
# - "Condemn": To publicly denounce and morally judge a target by invoking social norms or justice claims, positioning them as blameworthy and deserving of social sanction.
# - "Alienate": To cast a person or group as an out-group through dehumanizing, exclusionary framing that legitimizes social distance, rejection, or hostility.


# ==========================================
# 2. 媒体处理与工具函数 (保持像素级原样)
# ==========================================
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")


def get_video_duration_sec(video_path: str) -> float:
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
           video_path]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    return max(0.0, float(out))


def pick_4_frame_timestamps(duration: float, rng: random.Random) -> list:
    if duration <= 0: return [0.0] * 4
    eps, tail_margin = 1e-3, min(1.0, max(0.3, duration * 0.05))
    safe_end = max(0.0, duration - tail_margin)
    if safe_end <= eps: safe_end = duration
    bin_len = safe_end / 4.0
    ts = []
    for i in range(4):
        start, end = i * bin_len, min(safe_end, (i + 1) * bin_len)
        ts.append(max(0.0, min(safe_end, rng.uniform(start, max(start + eps, end - eps)))))
    return ts


def extract_video_frames_4(video_path: str, out_dir: Path, sample_id: str, seed: int) -> list:
    duration = get_video_duration_sec(video_path)
    rng = random.Random(seed + int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16))
    ts = pick_4_frame_timestamps(duration, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    def try_extract_one_frame(timestamp: float, out_path: Path) -> bool:
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-y", str(out_path)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0

    for idx, t in enumerate(ts):
        out_path = out_dir / f"{sample_id}_f{idx}.jpg"
        if out_path.exists():
            out_path.unlink()

        # Prefer one frame per quarter; if extraction fails, retry with fallback points in the same segment.
        seg_start = (duration / 4.0) * idx
        seg_end = min(duration, (duration / 4.0) * (idx + 1))
        seg_mid = (seg_start + seg_end) / 2.0
        seg_lo = max(0.0, seg_start + 1e-3)
        seg_hi = max(seg_lo, min(duration, seg_end - 1e-3))
        candidate_ts = [
            min(max(t, seg_lo), seg_hi),
            seg_mid,
            seg_lo,
            seg_hi
        ]
        # Add extra in-segment retries to improve success rate while keeping strict quarter locality.
        if seg_hi > seg_lo:
            for _ in range(3):
                candidate_ts.append(rng.uniform(seg_lo, seg_hi))

        # Deduplicate close timestamps while preserving order.
        dedup_ts = []
        seen = set()
        for c in candidate_ts:
            key = round(c, 3)
            if key not in seen:
                seen.add(key)
                dedup_ts.append(c)

        ok = False
        for c in dedup_ts:
            if try_extract_one_frame(c, out_path):
                frame_paths.append(str(out_path))
                ok = True
                break
            if out_path.exists():
                out_path.unlink()

        if not ok and out_path.exists():
            out_path.unlink()
    return frame_paths


def local_image_to_data_url(local_path: str) -> str:
    ext = Path(local_path).suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(Path(local_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def normalize_text(text):
    return str(text).strip().lower()


def normalize_slot(text):
    return str(text).strip().casefold()


def normalize_choice_key(text: str) -> str:
    s = normalize_text(text).replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def canonicalize_choice(raw_value: str, valid_choices: list) -> str:
    if not valid_choices:
        return normalize_text(raw_value)

    norm_to_canonical = {}
    stripped_to_canonical = {}
    for c in valid_choices:
        c_norm = normalize_choice_key(c)
        norm_to_canonical[c_norm] = c
        stripped_to_canonical[re.sub(r"[^a-z0-9]+", "", c_norm)] = c

    raw_norm = normalize_choice_key(raw_value)
    if raw_norm in norm_to_canonical:
        return norm_to_canonical[raw_norm]

    raw_stripped = re.sub(r"[^a-z0-9]+", "", raw_norm)
    if raw_stripped in stripped_to_canonical:
        return stripped_to_canonical[raw_stripped]
    return normalize_text(raw_value)


def canonicalize_slot_value(raw_value: str, kind: str) -> str:
    val = normalize_slot(raw_value)
    allowed = {f"{kind}{i}" for i in range(4)}
    if val in allowed:
        return val

    m = re.search(rf"{kind}\s*([0-3])", val)
    if m:
        return f"{kind}{m.group(1)}"

    return val


def build_response_format_for_situation(sit_norm: str) -> dict:
    mechs = VALID_MECHANISMS.get(sit_norm, [])
    labels = VALID_LABELS.get(sit_norm, [])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"evaluation_output_{sit_norm or 'unknown'}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "mechanism": {"type": "string", "enum": mechs},
                    "label": {"type": "string", "enum": labels},
                    "subject": {"type": "string", "enum": ["subject0", "subject1", "subject2", "subject3"]},
                    "target": {"type": "string", "enum": ["target0", "target1", "target2", "target3"]}
                },
                "required": ["mechanism", "label", "subject", "target"],
                "additionalProperties": False
            }
        }
    }


def build_situation_constraint_prompt(sit_norm: str) -> str:
    mechs = VALID_MECHANISMS.get(sit_norm, [])
    labels = VALID_LABELS.get(sit_norm, [])
    return (
        f'Current sample situation is "{sit_norm}". '
        f'You MUST choose mechanism ONLY from: {mechs}. '
        f'You MUST choose label ONLY from: {labels}. '
        f'Do not use labels or mechanisms from other situations.'
    )


# ==========================================
# 3. 核心评估流程
# ==========================================
def process_single_sample(sample, client):
    inp = sample['input']
    out = sample['output']
    sample_id = inp.get('id') or inp.get('samples_id')

    # 获取标准答案中的固定字段
    given_situation = out.get("situation", "")
    given_domain = out.get("domain", "")
    given_culture = out.get("culture", "")
    sit_norm = normalize_text(given_situation)
    if sit_norm not in VALID_MECHANISMS or sit_norm not in VALID_LABELS:
        return {"id": sample_id, "error": f"Unknown situation: {given_situation}"}

    subjects_raw = [out.get('subject', ''), out.get('subject1', ''), out.get('subject2', ''), out.get('subject3', '')]
    targets_raw = [out.get('target', ''), out.get('target1', ''), out.get('target2', ''), out.get('target3', '')]
    subjects_raw = [s for s in subjects_raw if str(s).strip()]
    targets_raw = [t for t in targets_raw if str(t).strip()]

    if len(subjects_raw) != 4 or len(targets_raw) != 4:
        return {"id": sample_id, "error": f"Choices not exactly 4", "original_sample": sample}

    rng = random.Random(RANDOM_SEED + int(hashlib.md5(str(sample_id).encode()).hexdigest()[:8], 16))
    rng.shuffle(subjects_raw)
    rng.shuffle(targets_raw)

    subject_slots = {f"subject{i}": subjects_raw[i] for i in range(4)}
    target_slots = {f"target{i}": targets_raw[i] for i in range(4)}
    true_subject_str = out.get("subject", "")
    true_target_str = out.get("target", "")
    true_subject_slot = next(
        (k for k, v in subject_slots.items() if normalize_slot(v) == normalize_slot(true_subject_str)),
        None
    )
    true_target_slot = next(
        (k for k, v in target_slots.items() if normalize_slot(v) == normalize_slot(true_target_str)),
        None
    )

    if true_subject_slot is None or true_target_slot is None:
        return {"id": sample_id, "error": "Ground-truth mapping error", "original_sample": sample}

    user_text = (
        f"Situation: {given_situation}\n"
        f"Domain: {given_domain}\n"
        f"Culture: {given_culture}\n"
        f"Text: {inp.get('text', '')}\n"
    )
    if 'audio_caption' in inp and inp['audio_caption']:
        user_text += f"Audio Caption: {inp['audio_caption']}\n"

    user_text += "\n\nSubject slot mapping:\n" + "\n".join([f"{k} = {v}" for k, v in subject_slots.items()])
    user_text += "\n\nTarget slot mapping:\n" + "\n".join([f"{k} = {v}" for k, v in target_slots.items()])
    user_text += "\n\nCRITICAL REMINDER: For 'subject' and 'target', DO NOT copy the text descriptions. You MUST output the exact keys (e.g., 'subject0', 'target2')."
    user_text += "\n\nProvide the 4-field JSON response."

    content_blocks = [{"type": "text", "text": user_text}]
    is_video = 'media_path_local' in inp or str(inp.get('path', '')).endswith('.mp4')
    try:
        if is_video:
            vid_path = inp.get('media_path_local') or inp.get('path')
            frames = extract_video_frames_4(vid_path, Path("./_temp_eval_frames"), sample_id, RANDOM_SEED)
            if len(frames) < 4:
                print(f"[FrameWarning] id={sample_id} extracted_frames={len(frames)}/4")
            for f in frames: content_blocks.append(
                {"type": "image_url", "image_url": {"url": local_image_to_data_url(f)}})
        else:
            img_path = inp.get('path')
            content_blocks.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(img_path)}})
    except Exception as e:
        return {"id": sample_id, "error": f"Media error: {e}"}

    prediction = None
    api_error = None
    raw_pred_mech, raw_pred_label, raw_pred_sub, raw_pred_tgt = "", "", "", ""
    constraint_prompt = build_situation_constraint_prompt(sit_norm)
    response_format = build_response_format_for_situation(sit_norm)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": constraint_prompt},
                    {"role": "user", "content": content_blocks}
                ],
                response_format=response_format
                #, temperature=0.0
            )
            if response.choices[0].message.content:
                prediction = json.loads(response.choices[0].message.content)
                # 记录原始生成的错误文本用于查错
                raw_pred_mech = prediction.get("mechanism", "")
                raw_pred_label = prediction.get("label", "")
                raw_pred_sub = prediction.get("subject", "")
                raw_pred_tgt = prediction.get("target", "")

                api_error = None
                break
            api_error = "Empty model response content"
        except Exception as e:
            api_error = str(e)
            time.sleep(2)

    if prediction is None:
        return {"id": sample_id, "error": api_error or "No valid prediction returned"}

    prediction["mechanism"] = canonicalize_choice(raw_pred_mech, VALID_MECHANISMS[sit_norm])
    prediction["label"] = canonicalize_choice(raw_pred_label, VALID_LABELS[sit_norm])
    prediction["subject"] = canonicalize_slot_value(raw_pred_sub, "subject")
    prediction["target"] = canonicalize_slot_value(raw_pred_tgt, "target")

    # 错误分析：基于给定的 situation 校验机制和标签是否合法
    mech_norm = normalize_choice_key(prediction["mechanism"])
    label_norm = normalize_choice_key(prediction["label"])
    allowed_mechs = [normalize_choice_key(m) for m in VALID_MECHANISMS.get(sit_norm, [])]
    allowed_labels = [normalize_choice_key(l) for l in VALID_LABELS.get(sit_norm, [])]

    error_analysis = {
        "mech_mismatch": mech_norm not in allowed_mechs if allowed_mechs else True,
        "label_mismatch": label_norm not in allowed_labels if allowed_labels else True,
        "subject_format_error": normalize_slot(prediction["subject"]) not in {"subject0", "subject1", "subject2", "subject3"},
        "target_format_error": normalize_slot(prediction["target"]) not in {"target0", "target1", "target2", "target3"},
        "predicted_mechanism": prediction["mechanism"], "predicted_label": prediction["label"],
        "predicted_subject": prediction["subject"], "predicted_target": prediction["target"],
        "raw_predicted_mechanism": raw_pred_mech, "raw_predicted_label": raw_pred_label,
        "raw_predicted_subject": raw_pred_sub, "raw_predicted_target": raw_pred_tgt
    }

    # 判定
    fields = ["mechanism", "label", "subject", "target"]
    matches = {}
    is_strict = True
    for f in fields:
        if f in ["subject", "target"]:
            m = (normalize_slot(prediction.get(f, "")) == (true_subject_slot if f == "subject" else true_target_slot))
        elif f == "mechanism":
            gt_val = canonicalize_choice(out.get(f, ""), VALID_MECHANISMS.get(sit_norm, []))
            pd_val = canonicalize_choice(prediction.get(f, ""), VALID_MECHANISMS.get(sit_norm, []))
            m = (normalize_choice_key(pd_val) == normalize_choice_key(gt_val))
        elif f == "label":
            gt_val = canonicalize_choice(out.get(f, ""), VALID_LABELS.get(sit_norm, []))
            pd_val = canonicalize_choice(prediction.get(f, ""), VALID_LABELS.get(sit_norm, []))
            m = (normalize_choice_key(pd_val) == normalize_choice_key(gt_val))
        else:
            m = (normalize_text(prediction.get(f, "")) == normalize_text(out.get(f, "")))
        matches[f] = m
        if not m: is_strict = False

    return {
        "id": sample_id,
        "ground_truth": {**{f: out.get(f) for f in fields}, "subject": true_subject_slot, "target": true_target_slot},
        "prediction": prediction, "matches": matches, "strict_match": is_strict, "error_analysis": error_analysis,
        "meta_situation": out.get("situation"), "meta_domain": out.get("domain"), "meta_culture": out.get("culture"),
        "slot_maps": {"subject_slots": subject_slots, "target_slots": target_slots,
                      "true_subject_slot": true_subject_slot, "true_target_slot": true_target_slot}
    }


# ==========================================
# 4. 指标计算与可视化 (4预测项 + Strict All柱子)
# ==========================================
def calculate_metrics_for_subset(results_list):
    if not results_list: return {"Count": 0}
    metrics = {"Count": len(results_list)}
    metrics["Strict_All4_Acc"] = accuracy_score([1] * len(results_list),
                                                [1 if r['strict_match'] else 0 for r in results_list])

    fields = ["mechanism", "label", "subject", "target"]
    for f in fields:
        y_true = []
        y_pred = []
        for r in results_list:
            sit_norm = normalize_text(r.get("meta_situation", ""))
            if f == "mechanism":
                gt = canonicalize_choice(r['ground_truth'].get(f, ""), VALID_MECHANISMS.get(sit_norm, []))
                pd = canonicalize_choice(r['prediction'].get(f, ""), VALID_MECHANISMS.get(sit_norm, []))
                y_true.append(normalize_choice_key(gt))
                y_pred.append(normalize_choice_key(pd))
            elif f == "label":
                gt = canonicalize_choice(r['ground_truth'].get(f, ""), VALID_LABELS.get(sit_norm, []))
                pd = canonicalize_choice(r['prediction'].get(f, ""), VALID_LABELS.get(sit_norm, []))
                y_true.append(normalize_choice_key(gt))
                y_pred.append(normalize_choice_key(pd))
            elif f == "subject":
                gt = normalize_slot(r['ground_truth'].get(f, ""))
                pd = canonicalize_slot_value(r['prediction'].get(f, ""), "subject")
                y_true.append(gt)
                if pd not in {"subject0", "subject1", "subject2", "subject3"}:
                    y_pred.append("invalid_format")
                else:
                    y_pred.append(pd)
            elif f == "target":
                gt = normalize_slot(r['ground_truth'].get(f, ""))
                pd = canonicalize_slot_value(r['prediction'].get(f, ""), "target")
                y_true.append(gt)
                if pd not in {"target0", "target1", "target2", "target3"}:
                    y_pred.append("invalid_format")
                else:
                    y_pred.append(pd)
            else:
                y_true.append(normalize_text(r['ground_truth'].get(f, "")))
                y_pred.append(normalize_text(r['prediction'].get(f, "")))
        metrics[f"{f}_Accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{f}_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return metrics


def plot_metrics(report, output_path, successful_count, model_name):
    metric_items = [
        ("Strict_All4_Acc", "Strict"),
        ("mechanism_Accuracy", "Mech Acc"),
        ("mechanism_F1", "Mech F1"),
        ("label_Accuracy", "Label Acc"),
        ("label_F1", "Label F1"),
        ("subject_Accuracy", "Subj Acc"),
        ("subject_F1", "Subj F1"),
        ("target_Accuracy", "Tgt Acc"),
        ("target_F1", "Tgt F1"),
    ]
    left_style_palette = [
        "#E8CFC5", "#E4C0B5", "#E9B2A3", "#EB9F8D",
        "#F08E79", "#F37F66", "#F26445", "#DE5A3F",
        "#C9563E", "#B3523B", "#9F4C37", "#8A4633"
    ]
    bg_color = "#ECECEC"
    grid_color = "#D0D0D0"
    edge_color = "#5E5954"
    plt.rcParams["font.family"] = "serif"

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_float(v):
        return float(v) if isinstance(v, (int, float)) else np.nan

    def _safe_count(v):
        return int(v) if isinstance(v, (int, float)) else 0

    def _find_overall_key(rep: dict) -> str:
        for k in rep.keys():
            if str(k).startswith("1. Overall"):
                return k
        return ""

    def _metric_values(stats: dict) -> list:
        return [_to_float(stats.get(k, np.nan)) for k, _ in metric_items]

    def _plot_overall(overall_stats: dict, save_path: Path):
        vals = _metric_values(overall_stats)
        labels = [x[1] for x in metric_items]
        x = np.arange(len(labels))

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        bars = ax.bar(
            x,
            np.nan_to_num(vals, nan=0.0),
            color=[left_style_palette[i % len(left_style_palette)] for i in range(len(labels))],
            edgecolor=edge_color,
            linewidth=0.8
        )
        ax.set_title(f"Overall Metrics ({model_name}, N={_safe_count(overall_stats.get('Count', successful_count))})",
                     fontsize=14, weight="bold", pad=12)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=grid_color, linewidth=0.85, alpha=0.9)
        ax.grid(axis="x", color=grid_color, linewidth=0.45, alpha=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        for spine in ax.spines.values():
            spine.set_color(edge_color)
            spine.set_linewidth(0.9)
        for i, bar in enumerate(bars):
            v = vals[i]
            txt = "-" if np.isnan(v) else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, txt, ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_grouped(section_key: str, section_title: str, save_path: Path):
        group_data = report.get(section_key, {})
        items = [(k, v) for k, v in group_data.items() if isinstance(v, dict)]
        labels = [x[1] for x in metric_items]
        x = np.arange(len(labels))

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        if not items:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{section_title}\nNo data", ha="center", va="center", fontsize=13)
        else:
            n = len(items)
            width = min(0.8 / max(n, 1), 0.18)
            offsets = (np.arange(n) - (n - 1) / 2.0) * width
            for idx, (name, stats) in enumerate(items):
                vals = _metric_values(stats)
                ax.bar(
                    x + offsets[idx],
                    np.nan_to_num(vals, nan=0.0),
                    width=width * 0.95,
                    color=left_style_palette[idx % len(left_style_palette)],
                    edgecolor=edge_color,
                    linewidth=0.65,
                    label=f"{name} (n={_safe_count(stats.get('Count', 0))})"
                )

            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.grid(axis="y", color=grid_color, linewidth=0.85, alpha=0.9)
            ax.grid(axis="x", color=grid_color, linewidth=0.45, alpha=0.45)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=2 if len(items) <= 6 else 3,
                frameon=False,
                fontsize=9
            )
            for spine in ax.spines.values():
                spine.set_color(edge_color)
                spine.set_linewidth(0.9)

        ax.set_title(section_title, fontsize=14, weight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _collect_table_rows(overall_key: str) -> list:
        rows = []
        overall_stats = report.get(overall_key, {})
        rows.append(("Overall", _safe_count(overall_stats.get("Count", 0)), _metric_values(overall_stats)))

        for s in SITUATIONS:
            stats = report.get("2. By Situation", {}).get(s)
            if isinstance(stats, dict):
                rows.append((f"Sit: {s}", _safe_count(stats.get("Count", 0)), _metric_values(stats)))
        for d in DOMAINS:
            stats = report.get("3. By Domain", {}).get(d)
            if isinstance(stats, dict):
                rows.append((f"Dom: {d}", _safe_count(stats.get("Count", 0)), _metric_values(stats)))
        for c in CULTURES:
            stats = report.get("4. By Culture", {}).get(c)
            if isinstance(stats, dict):
                rows.append((f"Cul: {c}", _safe_count(stats.get("Count", 0)), _metric_values(stats)))
        return rows

    def _plot_summary_table(rows: list, save_path: Path):
        col_labels = ["Group", "N"] + [x[1] for x in metric_items]
        table_text = []
        for group, n, vals in rows:
            row = [group, str(n)]
            row.extend(["-" if np.isnan(v) else f"{v:.3f}" for v in vals])
            table_text.append(row)

        fig_h = max(7.0, 0.43 * (len(table_text) + 2))
        fig, ax = plt.subplots(figsize=(22, fig_h))
        fig.patch.set_facecolor(bg_color)
        ax.axis("off")
        ax.set_title("Evaluation Metrics Summary Table", fontsize=15, weight="bold", pad=12)

        tbl = ax.table(cellText=table_text, colLabels=col_labels, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.25)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#F0E3DF")
                cell.set_text_props(weight="bold", color="#3A312A")
            else:
                cell.set_facecolor("#F7F4F3" if r % 2 == 0 else "#EFEAE8")
                if c == 0:
                    cell.set_text_props(ha="left")

        for r in range(len(table_text) + 1):
            tbl[(r, 0)].set_width(0.28)
            tbl[(r, 1)].set_width(0.05)
            for c in range(2, len(col_labels)):
                tbl[(r, c)].set_width(0.065)

        best_bg = "#D4E7CF"
        worst_bg = "#F1C7BE"
        for metric_col_idx in range(len(metric_items)):
            cidx = metric_col_idx + 2
            values = []
            for ridx, (_, n, vals) in enumerate(rows, start=1):
                v = vals[metric_col_idx]
                if n > 0 and not np.isnan(v):
                    values.append((ridx, v))
            if not values:
                continue
            best_v = max(v for _, v in values)
            worst_v = min(v for _, v in values)

            if best_v == worst_v:
                for ridx, v in values:
                    if v == best_v:
                        tbl[(ridx, cidx)].set_facecolor(best_bg)
                continue

            for ridx, v in values:
                if v == best_v:
                    tbl[(ridx, cidx)].set_facecolor(best_bg)
                elif v == worst_v:
                    tbl[(ridx, cidx)].set_facecolor(worst_bg)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_strict_table(rows: list, save_path: Path):
        col_labels = ["Group", "N", "ACC"]
        table_text = []
        for group, n, vals in rows:
            strict_v = vals[0] if vals else np.nan
            table_text.append([group, str(n), "-" if np.isnan(strict_v) else f"{strict_v:.3f}"])

        fig_h = max(7.0, 0.43 * (len(table_text) + 2))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        fig.patch.set_facecolor(bg_color)
        ax.axis("off")
        ax.set_title("Evaluation Strict-Only Table", fontsize=15, weight="bold", pad=12)

        tbl = ax.table(cellText=table_text, colLabels=col_labels, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.28)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#F0E3DF")
                cell.set_text_props(weight="bold", color="#3A312A")
            else:
                cell.set_facecolor("#F7F4F3" if r % 2 == 0 else "#EFEAE8")
                if c == 0:
                    cell.set_text_props(ha="left")

        # Make first column only as wide as needed for text content.
        max_group_len = max((len(str(group)) for group, _, _ in rows), default=10)
        group_w = min(0.54, max(0.38, 0.0125 * max_group_len + 0.10))
        n_w = 0.12
        acc_w = 0.16
        for r in range(len(table_text) + 1):
            tbl[(r, 0)].set_width(group_w)
            tbl[(r, 1)].set_width(n_w)
            tbl[(r, 2)].set_width(acc_w)

        valid = []
        for ridx, (_, n, vals) in enumerate(rows, start=1):
            v = vals[0] if vals else np.nan
            if n > 0 and not np.isnan(v):
                valid.append((ridx, v))
        if valid:
            best_v = max(v for _, v in valid)
            worst_v = min(v for _, v in valid)
            best_bg = "#D4E7CF"
            worst_bg = "#F1C7BE"
            if best_v == worst_v:
                for ridx, v in valid:
                    if v == best_v:
                        tbl[(ridx, 2)].set_facecolor(best_bg)
            else:
                for ridx, v in valid:
                    if v == best_v:
                        tbl[(ridx, 2)].set_facecolor(best_bg)
                    elif v == worst_v:
                        tbl[(ridx, 2)].set_facecolor(worst_bg)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    overall_key = _find_overall_key(report)
    if not overall_key:
        return {}

    saved = {
        "overall": output_dir / OUTPUT_PLOT_OVERALL_FILE.name,
        "situation": output_dir / OUTPUT_PLOT_SITUATION_FILE.name,
        "domain": output_dir / OUTPUT_PLOT_DOMAIN_FILE.name,
        "culture": output_dir / OUTPUT_PLOT_CULTURE_FILE.name,
        "table": output_dir / OUTPUT_TABLE_FILE.name,
        "strict_table": output_dir / OUTPUT_STRICT_TABLE_FILE.name,
    }

    rows = _collect_table_rows(overall_key)
    _plot_overall(report.get(overall_key, {}), saved["overall"])
    _plot_grouped("2. By Situation", "By Situation Metrics", saved["situation"])
    _plot_grouped("3. By Domain", "By Domain Metrics", saved["domain"])
    _plot_grouped("4. By Culture", "By Culture Metrics", saved["culture"])
    _plot_summary_table(rows, saved["table"])
    _plot_strict_table(rows, saved["strict_table"])
    return saved


# ==========================================
# 5. 主干逻辑
# ==========================================
def main():
    with open(IMAGE_JSON_PATH, 'r') as f:
        img_data = json.load(f)
    with open(VIDEO_JSON_PATH, 'r') as f:
        vid_data = json.load(f)
    combined = img_data + vid_data
    random.seed(RANDOM_SEED)
    sampled = random.sample(combined, min(SAMPLE_SIZE, len(combined)))
    client = OpenAI(api_key=OPENAI_API_KEY)

    results, failed = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, s, client): s for s in sampled}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                res = f.result()
            except Exception as e:
                failed.append({"id": None, "error": f"Unhandled worker error: {e}"})
                print(f"[Failed] Unhandled worker error: {e}")
                if i % 10 == 0: print(f"Progress: {i}/{len(sampled)}")
                continue
            if "error" not in res:
                results.append(res)
            else:
                failed.append(res); print(f"[Failed] {res['error']}")
            if i % 10 == 0: print(f"Progress: {i}/{len(sampled)}")

    with open(OUTPUT_DETAILED_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_FAILED_FILE, 'w') as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    if not results: return
    report = {"1. Overall": calculate_metrics_for_subset(results)}
    for group_name, key in [("2. By Situation", "meta_situation"), ("3. By Domain", "meta_domain"),
                            ("4. By Culture", "meta_culture")]:
        report[group_name] = {val: calculate_metrics_for_subset([r for r in results if r[key] == val]) for val in
                              globals()[group_name.split()[-1].upper() + 'S']}

    # 错误分析报告
    err_rep = {"mechanism_mismatches": {"count": 0, "samples": []}, "label_mismatches": {"count": 0, "samples": []},
               "format_errors": {"count": 0, "samples": []}}
    for r in results:
        ea = r["error_analysis"]
        if ea["mech_mismatch"]:
            err_rep["mechanism_mismatches"]["count"] += 1
            err_rep["mechanism_mismatches"]["samples"].append(
                {"id": r["id"], "given_sit": r["meta_situation"], "pred_mech": ea["predicted_mechanism"]})
        if ea["label_mismatch"]:
            err_rep["label_mismatches"]["count"] += 1
            err_rep["label_mismatches"]["samples"].append(
                {"id": r["id"], "given_sit": r["meta_situation"], "pred_label": ea["predicted_label"]})
        if ea["subject_format_error"] or ea["target_format_error"]:
            err_rep["format_errors"]["count"] += 1
            err_rep["format_errors"]["samples"].append(
                {"id": r["id"], "pred_sub": ea["predicted_subject"], "pred_tgt": ea["predicted_target"]})

    report[f"1. Overall ({len(results)} Successful Samples)"] = report.pop("1. Overall")
    report["5. Error Analysis"] = err_rep

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    saved_artifacts = plot_metrics(report, OUTPUT_DIR, len(results), MODEL_NAME)
    print(f"Done! Strict Acc: {report[f'1. Overall ({len(results)} Successful Samples)']['Strict_All4_Acc']:.2%}")
    if saved_artifacts:
        print("Saved visualization files:")
        for p in saved_artifacts.values():
            print(f"- {p}")


if __name__ == "__main__":
    main()
