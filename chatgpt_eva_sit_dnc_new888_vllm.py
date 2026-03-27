import os
import json
import time
import random
import base64
import hashlib
import shutil
import subprocess
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").strip()
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY").strip()
MODEL_NAME = os.environ.get("EVAL_MODEL_NAME", "llava-1.5-7b-local").strip()
INPUT_JSON_PATH = os.environ.get(
    "INPUT_JSON_PATH",
    "/scratch/users/nus/e1553307/experiment/Data/review_assign_eval300_mixed_seed42.json",
).strip()
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "300"))
RANDOM_SEED = 42
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
MAX_RETRIES = 3
VIDEO_NUM_FRAMES = max(1, min(4, int(os.environ.get("VIDEO_NUM_FRAMES", "4"))))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "512"))

SCENARIOS = ["affection", "attitude", "intent"]
DOMAINS = ["Online & Social Media", "Public & Service", "Workplace", "Intimate Relationships", "Family Conversations",
           "Friend Group", "Education & Campus", "Friendship Interactions"]
CULTURES = ["General Culture", "Arab Culture", "American Culture", "Muslim Culture", "African American Culture",
            "Jewish Culture", "Indian Culture", "East Asian Culture"]

# ==========================================
# 动态创建输出文件夹 (对齐 Qwen 命名风格)
# ==========================================
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/scratch/users/nus/e1553307/experiment/results").strip())
OUTPUT_TAG = os.environ.get("OUTPUT_TAG", "").strip()
if OUTPUT_TAG:
    output_dir_name = OUTPUT_TAG
else:
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in MODEL_NAME)
    output_dir_name = f"{safe_model}_{SAMPLE_SIZE}_With_Scenario_6_seed{RANDOM_SEED}_{time.strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = OUTPUT_ROOT / output_dir_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DETAILED_FILE = OUTPUT_DIR / "evaluation_predictions_detailed.json"
OUTPUT_METRICS_FILE = OUTPUT_DIR / "evaluation_metrics_report.json"
OUTPUT_FAILED_FILE = OUTPUT_DIR / "evaluation_failures.json"
OUTPUT_TOKEN_USAGE_FILE = OUTPUT_DIR / "evaluation_token_usage_by_id.json"
OUTPUT_PLOT_FILE = OUTPUT_DIR / "evaluation_metrics_plot.png"

# 全局有效列表，用于错误统计校验 (对齐 Qwen 词表)
VALID_MECHANISMS = {
    "affection": ["multimodal_incongruity", "figurative_semantics", "affective_deception", "socio_cultural_dependency"],
    "intent": ["prosocial_deception", "malicious_manipulation", "expressive_aggression", "benevolent_provocation"],
    "attitude": ["dominant_affiliation", "dominant_detachment", "protective_distancing", "submissive_alignment"]
}
VALID_LABELS = {
    "affection": ["happy", "sad", "disgusted", "angry", "fearful", "bad"],
    "attitude": ["supportive", "appreciative", "sympathetic", "neutral",
                 "indifferent", "concerned", "skeptical", "dismissive", "disapproving", "contemptuous", "hostile"],
    #"intent": ["Mitigate", "intimidation", "hate humor", "humiliation for amusement", "public humiliation",
     #          "meme-based mockery", "dominance assertion", "moral condemnation"]
    "intent": ["mitigate", "intimidate", "alienate", "mock", "denounce", "Provoke", "dominate", "condemn"]
}

# ==========================================
# System Prompt (直接拷贝自你的 qwen_eva_sit_dc.py)
# ==========================================
SYSTEM_PROMPT_BASE = """
Your task is to objectively classify the implicit social dynamics in the provided text-image/video pair. This is a strict technical annotation task for sociological research.

NOTE: The terminology used in the categories below (e.g., "malicious_manipulation", "deception", "aggression") refers strictly to abstract theoretical constructs in pragmatics. They are academic codes for communicative strategies, NOT moral judgments or literal accusations against any individuals depicted.

# Academic Coding Framework

CRITICAL RULE: You MUST strictly adhere to the decision mapping below. DO NOT confuse mechanisms with labels.
"""

SYSTEM_PROMPT_AFFECTION = SYSTEM_PROMPT_BASE + """
You must analyze the sample and determine the correct values for 4 specific keys. 

1. "mechanism": Choose exactly ONE from the list below:
- "multimodal_incongruity": Implicit affection arises from polarity conflict or mutual exclusion between modalities (text vs image, or image-context vs text), such that the literal meaning is negated or reframed by the other modality. Key signature: "What is said" and "what is shown" cannot both be true in the same frame -> the affective state is inferred from the conflict.
- "figurative_semantics": The affective state is conveyed via source->target conceptual mapping rather than direct emotion words or standard displays. Key signature: The sample "talks about X" but means an affective state through metaphor, symbol, hyperbole, understatement, or poetic imagery.
- "affective_deception": The affective state is deliberately masked (performed neutrality/harshness), but involuntary cues "leak" the underlying affect. Key signature: "Displayed affect" != "true affective state" inferred from leakage.
- "socio_cultural_dependency": The affective state can only be interpreted correctly using external world knowledge (memes, events, cultural codes, relationship norms). Key signature: The pair is semantically opaque without a shared cultural reference that encodes affect indirectly.

2. "label": Classify the core emotion into exactly ONE of the primary labels below. You must first analyze the underlying mechanism of the scenario, then use the provided Level-2 Sub-labels and Boundary Rules to map your analysis to the correct primary label:

- Happy
  * Level-2 Sub-labels: Playful, Content, Interested, Proud, Accepted, Powerful, Peaceful, Trusting, Optimistic.
  * Core Mechanism: A positive state involving satisfaction, joy, or confidence, usually accompanied by positive evaluation of the current scenario (e.g., feeling accepted or hopeful).
  * Boundary Rule: If the expression is primarily factual with extremely weak emotion, classify as 'Neutral'. If the core mechanism is gratitude/thankfulness for someone's help, it strictly falls under 'Happy'.
- Sad
  * Level-2 Sub-labels: Lonely, Vulnerable, Despair, Guilty, Depressed, Hurt.
  * Core Mechanism: A negative state associated with loss, helplessness, or relationship setbacks, often leading to withdrawal or lack of motivation.
  * Boundary Rule: If the focus is on self-blame/compensation for a mistake, map to the 'Guilty' sub-label. If the focus is relational pain caused by others, map to the 'Hurt' sub-label. 
- Disgusted
  * Level-2 Sub-labels: Repelled, Awful, Disappointed, Disapproving.
  * Core Mechanism: A strong aversion or rejection toward physical stimuli (smell/food) or social/moral stimuli (immorality, hypocrisy, offense).
  * Boundary Rule: The core tendency is "AWAY FROM" (want to avoid/reject/deny). If the emotion primarily involves blame, confrontation, or emphasizing "the other party is wrong" rather than mere aversion, it leans towards 'Angry'.
- Angry
  * Level-2 Sub-labels: Let down, Humiliated, Bitter, Mad, Aggressive, Frustrated, Distant, Critical.
  * Core Mechanism: Triggered by being offended, hindered, or treated unfairly, leading to hostility, frustration, and an intention to control, fight back, or argue.
  * Boundary Rule: The core tendency is "AGAINST". If the expression is merely cold or "I don't care" (without confrontation), it leans towards 'Bad' or 'Neutral'. If it's primarily aversion without the urge to fight back, it leans towards 'Disgusted'.
- Fearful
  * Level-2 Sub-labels: Scared, Anxious, Insecure, Weak, Rejected, Threatened.
  * Core Mechanism: Centered around threat and insecurity, expecting that "something bad might happen / I might get hurt," leading to a tendency to avoid, seek protection, or increase vigilance.
  * Boundary Rule: Includes continuous worry caused by long-term uncertainty ('Anxious'). If it's just a brief reaction to unexpected factual info without a core sense of threat, map it to 'Bad' or 'Neutral' based on context.
- Bad
  * Level-2 Sub-labels: Bored, Busy, Stressed, Tired.
  * Core Mechanism: A generalized negative state for scenarios that do not fit the specific intensity or clear triggers of Sad, Angry, Disgusted, or Fearful. It includes feeling uncomfortable, exhausted, apathetic, or experiencing cognitive dissonance.
  * Boundary Rule: Use this category for low-intensity negative states or mixed/cynical contexts. For instance, implicit scenarios involving "dark humor" (地狱笑话) should be classified here. The mechanism of dark humor relies on taboo, tragedy, or moral discomfort, which aligns with this generalized uncomfortable/apathetic state, rather than genuine positive joy (Happy) or targeted hostility (Angry).
- Neutral
  * Boundary Rule: Use this ONLY if the expression is primarily factual with zero or extremely weak emotional intensity, as directed by the boundary rules of other labels.

3. "subject": You MUST select EXACTLY ONE string from this strict list: ["subject0", "subject1", "subject2", "subject3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

4. "target": You MUST select EXACTLY ONE string from this strict list: ["target0", "target1", "target2", "target3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

# Output Format Requirements
Output ONLY a strict JSON object with EXACTLY 4 keys. You MUST adhere to the exact values requested. Use this exact schema:
{
  "mechanism": "<insert ONE valid mechanism here>",
  "label": "<insert ONE valid label here>",
  "subject": "<MUST be subject0, subject1, subject2, or subject3>",
  "target": "<MUST be target0, target1, target2, or target3>"
}
Do NOT write markdown blocks (no ```json).
Do NOT include any explanations, rationales, or chain-of-thought reasoning.
"""



SYSTEM_PROMPT_ATTITUDE = SYSTEM_PROMPT_BASE + """
You must analyze the sample and determine the correct values for 4 specific keys. 

1. "mechanism": Choose exactly ONE from the list below:
- "dominant_affiliation": Surface friendliness or acceptance is used to assert a "stronger to weaker" superiority; closeness is granted from above. Key signatures include: Talking Down (treating the target like a child to establish superiority), Patronizing Praise (praising while implicitly lowering the standard for the target), or Benevolent Control (depriving the target's autonomy under the guise of "for your own good").
- "dominant_detachment": Establishing high status while cutting off emotional connection. Key signatures include: Invalidation (standing on a moral/intellectual high ground to deny the target's value or feelings), Character Blaming (attributing the target's plight to their own moral/personality flaws; "they asked for it"), or Rationalizing Over (using cold logic to dismiss the target's perspective as unworthy of serious attention).
- "protective_distancing": A lower-power or vulnerable stance avoiding direct confrontation via non-commitment or psychological isolation. Key signatures include: Keeping It Open (suspending commitment/action without explicitly agreeing or disagreeing), Emotional Withdrawal (receiving information but noticeably dropping emotional engagement), or Skeptical Distance (appearing open but implicitly refusing to accept the target's premises).
- "submissive_alignment": Proactively lowering one's own status to seek safety, attachment, or protection. Key signatures include: Self-Diminishment (belittling oneself to gain acceptance), Over-Accommodation (sacrificing one's own stance to avoid conflict/rejection), or Leaning On (showing weakness to signal a need for the target's protection or decision-making).

2. "label": Classify the implicit attitude into EXACTLY ONE of the labels below based on your mechanism analysis. Use the Anchor Examples contextually and strictly apply the Boundary Rules:

- "Supportive"
  * Core Definition: Explicitly defends the target or affirms their legitimacy and value.
  * Anchor Examples (Non-exhaustive): e.g., making excuses for their failure, protecting their reputation, taking their side.
  * Boundary Rule: The focus is on active alignment/defense. If it only evaluates output positively without defending or taking a side, map to 'Appreciative'.
- "Appreciative"
  * Core Definition: Positive evaluation of the target's abilities, qualities, or achievements.
  * Anchor Examples (Non-exhaustive): e.g., praising effort, complimenting results or design.
  * Boundary Rule: The focus is on merit evaluation. It differs from 'Supportive' as it doesn't necessarily involve defending the target against adversity.
- "Sympathetic"
  * Core Definition: Empathy and understanding for the target's unfavorable scenario.
  * Anchor Examples (Non-exhaustive): e.g., emphasizing harsh environments, bad luck, or high difficulty to comfort and downplay responsibility.
  * Boundary Rule: The focus is on shared suffering/excusing circumstances. If the subject actively defends the target's actions as fully correct, it leans towards 'Supportive'.
- "Neutral"
  * Core Definition: No clear value judgment or affective stance.
  * Anchor Examples (Non-exhaustive): e.g., objective statements of facts, news-like reports.
  * Boundary Rule: Use ONLY if the expression is purely factual. If there is a deliberate, cynical withdrawal of care, use 'Indifferent'.
- "Indifferent"
  * Core Definition: Explicitly conveying a lack of care or engagement regarding the target.
  * Anchor Examples (Non-exhaustive): e.g., "whatever," "doesn't matter," showing apathy.
  * Boundary Rule: The focus is on apathy. If the subject actively downplays the target's importance to shut them down, use 'Dismissive'.
- "Disapproving"
  * Core Definition: Negative evaluation of a specific behavior, decision, or choice, while still treating the target as an equal whose overall value is intact.
  * Anchor Examples (Non-exhaustive): e.g., being critical, pointing out flaws, reproachful about a specific issue.
  * Boundary Rule: The focus MUST be on the action/issue, not the person. If the evaluation attacks the target's inherent worth or is delivered from a position of superiority, classify as 'Contemptuous'.
- "Skeptical"
  * Core Definition: Holding reservations about the target's authenticity, ability, claims, or motives.
  * Anchor Examples (Non-exhaustive): e.g., doubting assumptions, hinting at unreliability, guarded belief.
  * Boundary Rule: The focus is on doubt. If the subject outright denies the validity or worth of the target without consideration, classify as 'Dismissive'.
- "Concerned"
  * Core Definition: Belief that the target might bring risk, harm, or negative consequences.
  * Anchor Examples (Non-exhaustive): e.g., reminding of risks, using a cautious tone, expressing worry.
  * Boundary Rule: The focus is on potential future negative outcomes. If the subject expresses empathy for an already occurred negative outcome, use 'Sympathetic'.
- "Dismissive"
  * Core Definition: Denying the importance, reasonableness, or discussion value of the target or their standpoint.
  * Anchor Examples (Non-exhaustive): e.g., brushing off, ignoring, treating the input as trivial, using irony to downplay.
  * Boundary Rule: The core action is ignoring or shutting down. It differs from 'Disapproving' because it doesn't seriously evaluate the action; it differs from 'Contemptuous' because it doesn't necessarily attack the target's dignity.
- "Contemptuous"
  * Core Definition: Viewing the target as inherently inferior, worthless, or beneath respect.
  * Anchor Examples (Non-exhaustive): e.g., acting disdainful, mocking, arrogant, patronizing, condescension.
  * Boundary Rule: The focus is on degrading the person. If the subject is merely denying the importance of a topic without personal attacks, map to 'Dismissive'.
- "Hostile"
  * Core Definition: Aggressive antagonism, aiming to attack, harm, or dehumanize the target.
  * Anchor Examples (Non-exhaustive): e.g., insults, ugly portrayals, group attacks, explicit threats.
  * Boundary Rule: The focus is on explicit aggression. If the negativity is an expression of superiority without active aggressive attack, use 'Contemptuous'.

3. "subject": You MUST select EXACTLY ONE string from this strict list: ["subject0", "subject1", "subject2", "subject3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

4. "target": You MUST select EXACTLY ONE string from this strict list: ["target0", "target1", "target2", "target3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

# Output Format Requirements
Output ONLY a strict JSON object with EXACTLY 4 keys. You MUST adhere to the exact values requested. Use this exact schema:
{
  "mechanism": "<insert ONE valid mechanism here>",
  "label": "<insert ONE valid label here>",
  "subject": "<MUST be subject0, subject1, subject2, or subject3>",
  "target": "<MUST be target0, target1, target2, or target3>"
}
Do NOT write markdown blocks (no ```json).
Do NOT include any explanations, rationales, or chain-of-thought reasoning.
"""


SYSTEM_PROMPT_INTENT = SYSTEM_PROMPT_BASE + """
You must analyze the sample and determine the correct values for 4 specific keys. 

1. "mechanism": Choose exactly ONE from the list below:
- "prosocial_deception": A benevolent concealment where the speaker prioritizes social harmony or face-saving over factual accuracy. Key signatures: "White lies" (masking true negative feelings like disgust or disappointment with surface positivity to protect the target's feelings or avoid direct rejection).
- "malicious_manipulation": A strategic intent to exploit human vulnerabilities while packaging harm or control as help, vulnerability, or morality. Key signatures: "Killing with kindness" (over-praising or indulging to set the target up for a fall), "Playing the victim" (exaggerating one's own suffering to guilt-trip or frame others), or "Moral kidnapping" (using high moral standards to force compliance or stigmatize the target).
- "expressive_aggression": Indirect or veiled hostility used to attack, mock, or control the target while maintaining a facade. Key signatures: "Veiled abuse" (publicly mocking or insulting under the guise of giving advice/gifts) or "Implicit threats" (using overly specific observations or fake care to instill psychological pressure and fear).
- "benevolent_provocation": A strategic disguise used to trigger a desired action or reveal the truth. Key signatures: "Goading" or "Reverse psychology" (creating a fake scenario, pretending ignorance, or challenging the target to force them to expose their true stance or confess).

2. "label": Classify the core intent into EXACTLY ONE of the labels below based on your mechanism analysis. Use the Anchor Examples contextually and strictly apply the Boundary Rules:

- "Mitigate"
  * Core Definition: Intent to de-escalate tension, save face, and maintain a cooperative relationship.
  * Anchor Examples (Non-exhaustive): e.g., offering a compromise, smoothing over an awkward scenario, telling a prosocial "white lie".
  * Boundary Rule: The primary goal MUST be reducing friction or protecting harmony. If the subject uses fake empathy merely to set up an attack later, classify as 'Provoke' or 'Dominate'.
- "Intimidate"
  * Core Definition: Intent to coerce the target through fear, implied consequences, or threats.
  * Anchor Examples (Non-exhaustive): e.g., implicit threats, leveraging specific knowledge to instill psychological pressure, forcing compliance.
  * Boundary Rule: The focus is on instilling fear for control. If the aggression is purely for amusement without the goal of coercing behavior, classify as 'Mock'.
- "Alienate"
  * Core Definition: Intent to isolate the target by stripping them of equal status and reducing them to an "outsider" or negative stereotype.
  * Anchor Examples (Non-exhaustive): e.g., othering ("us vs. them"), dehumanizing, group-based exclusion.
  * Boundary Rule: The focus is on identity/group exclusion. If the public isolation is based on a specific moral failing rather than identity/stereotyping, map to 'Denounce'.
- "Mock"
  * Core Definition: Intent to entertain oneself or others at the target's expense by making them look foolish.
  * Anchor Examples (Non-exhaustive): e.g., treating flaws as a joke, humiliation for amusement, teasing.
  * Boundary Rule: The primary goal is amusement. If the humor/mockery is used as a veil for a severe attack or baiting to trigger anger, classify as 'Provoke'. If the goal is strictly asserting power, map to 'Dominate'.
- "Denounce"
  * Core Definition: Intent to socially punish the target by exposing their actions to an audience and rallying social pressure.
  * Anchor Examples (Non-exhaustive): e.g., public shaming, canceling, rallying a crowd against the target.
  * Boundary Rule: MUST involve a public or multi-party audience dimension. If the judgment is purely one-on-one based on rules/ethics without rallying an audience, classify as 'Condemn'.
- "Provoke"
  * Core Definition: Intent to bait, anger, or attack the target while maintaining plausible deniability, often using disguised hostility.
  * Anchor Examples (Non-exhaustive): e.g., using meme-based mockery, heavy irony, reverse psychology, goading to force a reaction.
  * Boundary Rule: The focus is on triggering a reaction or attacking from behind a shield of ambiguity. If the hostility is direct and aims to establish a clear hierarchy, map to 'Dominate'.
- "Dominate"
  * Core Definition: Intent to establish power or intellectual asymmetry, forcing the target into a subordinate position.
  * Anchor Examples (Non-exhaustive): e.g., pulling rank, acting patronizingly, condescension, "killing with kindness".
  * Boundary Rule: The focus is on establishing a vertical hierarchy ("I am above you"). If the control is achieved primarily through fear rather than status assertion, map to 'Intimidate'.
- "Condemn"
  * Core Definition: Intent to judge the target's actions as ethically wrong or invalid from a moral or logical high ground.
  * Anchor Examples (Non-exhaustive): e.g., playing the righteous judge, moral kidnapping, ethical lecturing.
  * Boundary Rule: The focus is strictly on rules and ethics. It differs from 'Denounce' because it doesn't necessarily rely on rallying a public mob, and differs from 'Dominate' because the superiority stems from a moral code rather than pure social rank.

3. "subject": You MUST select EXACTLY ONE string from this strict list: ["subject0", "subject1", "subject2", "subject3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

4. "target": You MUST select EXACTLY ONE string from this strict list: ["target0", "target1", "target2", "target3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

# Output Format Requirements
Output ONLY a strict JSON object with EXACTLY 4 keys. You MUST adhere to the exact values requested. Use this exact schema:
{
  "mechanism": "<insert ONE valid mechanism here>",
  "label": "<insert ONE valid label here>",
  "subject": "<MUST be subject0, subject1, subject2, or subject3>",
  "target": "<MUST be target0, target1, target2, or target3>"
}
Do NOT write markdown blocks (no ```json).
Do NOT include any explanations, rationales, or chain-of-thought reasoning.
"""

# ==========================================
# 2. 媒体处理与工具函数 (保持像素级原貌)
# ==========================================
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
FFMPEG_AVAILABLE = shutil.which(FFMPEG_BIN) is not None
LOCAL_MEDIA_INDEX_JSON = os.environ.get(
    "LOCAL_MEDIA_INDEX_JSON",
    "/scratch/users/nus/e1553307/experiment/Data/review_assign_all_dedup_unique_localpaths.json",
).strip()


def load_local_media_index(index_path: str) -> dict:
    if not index_path:
        return {}
    p = Path(index_path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return {}
    if isinstance(loaded, dict) and isinstance(loaded.get("data"), list):
        loaded = loaded["data"]
    if not isinstance(loaded, list):
        return {}

    media_by_id = {}
    for sample in loaded:
        if not isinstance(sample, dict):
            continue
        inp = sample.get("input", {})
        if not isinstance(inp, dict):
            continue
        media = inp.get("media", {})
        if not isinstance(media, dict):
            media = {}
        sid = sample.get("id") or inp.get("id") or inp.get("samples_id")
        pth = (
            inp.get("media_path_local")
            or inp.get("path")
            or media.get("video_path_local")
            or media.get("video_path")
            or media.get("image_path")
        )
        if sid and pth:
            media_by_id[str(sid)] = str(pth)
    return media_by_id


LOCAL_MEDIA_BY_ID = load_local_media_index(LOCAL_MEDIA_INDEX_JSON)


def get_video_duration_sec(video_path: str) -> float:
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
           video_path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return max(0.0, float(out))
    except Exception:
        # Fallback for environments where ffprobe is not available.
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps <= 0.0 or frame_count <= 0.0:
            return 0.0
        return max(0.0, float(frame_count / fps))


def pick_frame_timestamps(duration: float, num_frames: int, rng: random.Random) -> list:
    num_frames = max(1, int(num_frames))
    if duration <= 0:
        return [0.0] * num_frames
    eps, tail_margin = 1e-3, min(1.0, max(0.3, duration * 0.05))
    safe_end = max(0.0, duration - tail_margin)
    if safe_end <= eps:
        safe_end = duration
    bin_len = safe_end / float(num_frames)
    ts = []
    for i in range(num_frames):
        start, end = i * bin_len, min(safe_end, (i + 1) * bin_len)
        ts.append(max(0.0, min(safe_end, rng.uniform(start, max(start + eps, end - eps)))))
    return ts


def extract_video_frames(video_path: str, out_dir: Path, sample_id: str, seed: int, num_frames: int) -> list:
    num_frames = max(1, int(num_frames))
    duration = get_video_duration_sec(video_path)
    rng = random.Random(seed + int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16))
    ts = pick_frame_timestamps(duration, num_frames, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    def try_extract_one_frame_cv2(timestamp: float, out_path: Path) -> bool:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return False
            ok = cv2.imwrite(str(out_path), frame)
            return bool(ok) and out_path.exists() and out_path.stat().st_size > 0
        finally:
            cap.release()

    def try_extract_one_frame(timestamp: float, out_path: Path) -> bool:
        if not FFMPEG_AVAILABLE:
            return try_extract_one_frame_cv2(timestamp, out_path)
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            "-y", str(out_path)
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True
        return try_extract_one_frame_cv2(timestamp, out_path)

    for idx, t in enumerate(ts):
        out_path = out_dir / f"{sample_id}_f{idx}.jpg"
        if out_path.exists():
            out_path.unlink()

        # Prefer one frame per quarter; if extraction fails, retry with fallback points in the same segment.
        seg_start = (duration / float(num_frames)) * idx
        seg_end = min(duration, (duration / float(num_frames)) * (idx + 1))
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
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(Path(local_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_media_path(sample_id, inp: dict) -> str:
    candidates = []
    p0 = inp.get("media_path_local")
    p1 = inp.get("path")
    if p0:
        candidates.append(str(p0))
    if p1:
        candidates.append(str(p1))

    for p in candidates:
        if Path(p).exists():
            return p

    sid = str(sample_id) if sample_id is not None else ""
    mapped = LOCAL_MEDIA_BY_ID.get(sid)
    if mapped and Path(mapped).exists():
        return mapped

    return candidates[0] if candidates else ""


def normalize_text(text):
    return str(text).strip().lower()


def normalize_slot(text):
    return str(text).strip().casefold()


def _pick_three_distractors(options_list, gt_value):
    if not isinstance(options_list, list):
        raise ValueError("options list is missing")

    gt_norm = normalize_slot(gt_value)
    deduped = []
    seen = set()
    for opt in options_list:
        opt_str = str(opt).strip()
        if not opt_str:
            continue
        key = normalize_slot(opt_str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opt_str)

    distractors = [x for x in deduped if normalize_slot(x) != gt_norm]
    if len(distractors) < 3:
        raise ValueError("Not enough valid distractors after removing ground truth")
    return distractors[:3]


def convert_new_format_to_legacy(sample):
    """
    Convert one item from new-format JSON into the legacy structure used by this evaluator.
    """
    if not isinstance(sample, dict):
        raise ValueError("sample must be a dict")

    inp_new = sample.get("input", {})
    if not isinstance(inp_new, dict):
        inp_new = {}
    media = inp_new.get("media", {})
    if not isinstance(media, dict):
        media = {}
    options = sample.get("options", {})
    if not isinstance(options, dict):
        options = {}
    gt = sample.get("ground_truth", {})
    if not isinstance(gt, dict):
        gt = {}
    diversity = sample.get("diversity", {})
    if not isinstance(diversity, dict):
        diversity = {}

    sample_id = sample.get("id") or inp_new.get("id") or inp_new.get("samples_id")
    scenario = inp_new.get("scenario", "")
    text = inp_new.get("text", "")
    audio_caption = media.get("audio_caption", "")

    video_path = str(media.get("video_path") or media.get("video_path_local") or "").strip()
    image_path = str(media.get("image_path") or "").strip()
    media_path = video_path or image_path

    if not sample_id:
        raise ValueError("missing sample id")
    if not media_path:
        raise ValueError("missing media path")

    gt_subject = gt.get("subject", "")
    gt_target = gt.get("target", "")
    subject_distractors = _pick_three_distractors(options.get("subject", []), gt_subject)
    target_distractors = _pick_three_distractors(options.get("target", []), gt_target)

    legacy_input = {
        "id": sample_id,
        "text": text,
        "path": media_path,
    }
    if video_path:
        legacy_input["media_path_local"] = video_path
    if audio_caption:
        legacy_input["audio_caption"] = audio_caption
    if media.get("video_url"):
        legacy_input["url"] = media.get("video_url")
    elif media.get("image_url"):
        legacy_input["url"] = media.get("image_url")

    legacy_output = {
        "scenario": scenario,
        "domain": diversity.get("domain", ""),
        "culture": diversity.get("culture", ""),
        "mechanism": gt.get("mechanism", ""),
        "label": gt.get("label", ""),
        "subject": gt_subject,
        "subject1": subject_distractors[0],
        "subject2": subject_distractors[1],
        "subject3": subject_distractors[2],
        "target": gt_target,
        "target1": target_distractors[0],
        "target2": target_distractors[1],
        "target3": target_distractors[2],
    }
    return {"input": legacy_input, "output": legacy_output}


def adapt_samples_to_legacy(data):
    if not data:
        return []

    first = data[0]
    is_new_format = (
        isinstance(first, dict)
        and "ground_truth" in first
        and "options" in first
        and isinstance(first.get("input"), dict)
    )
    if not is_new_format:
        return data

    converted = []
    skipped = 0
    for idx, sample in enumerate(data):
        try:
            converted.append(convert_new_format_to_legacy(sample))
        except Exception as e:
            skipped += 1
            sid = sample.get("id", f"idx_{idx}") if isinstance(sample, dict) else f"idx_{idx}"
            print(f"[LoadWarning] skip sample {sid}: {e}", flush=True)
    print(
        f"[LoadInfo] detected new-format input: total={len(data)} converted={len(converted)} skipped={skipped}",
        flush=True,
    )
    return converted


def extract_usage_from_response(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

    cached_tokens = 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


CONTEXT_OVERFLOW_ERROR_MARKERS = (
    "max_tokens must be at least 1",
    "maximum context length",
    "context length exceeded",
    "prompt is too long",
    "input is too long",
)


def is_context_overflow_error(err_msg: str) -> bool:
    txt = normalize_text(err_msg)
    return any(marker in txt for marker in CONTEXT_OVERFLOW_ERROR_MARKERS)


# ==========================================
# 3. 核心评估流程 (对齐 Qwen：注入 Scenario, Domain, Culture + 错误统计)
# ==========================================
def process_single_sample(sample, client):
    inp = sample['input']
    out = sample['output']
    sample_id = inp.get('id') or inp.get('samples_id')
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

    # 获取标准答案中的固定字段
    given_scenario = out.get("scenario", out.get("situation", ""))
    given_domain = out.get("domain", "")
    given_culture = out.get("culture", "")

    subjects_raw = [out.get('subject', ''), out.get('subject1', ''), out.get('subject2', ''), out.get('subject3', '')]
    targets_raw = [out.get('target', ''), out.get('target1', ''), out.get('target2', ''), out.get('target3', '')]
    subjects_raw = [s for s in subjects_raw if str(s).strip()]
    targets_raw = [t for t in targets_raw if str(t).strip()]

    if len(subjects_raw) != 4 or len(targets_raw) != 4:
        return {"id": sample_id, "error": f"Choices not exactly 4", "original_sample": sample, "token_usage": token_usage}

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
        return {"id": sample_id, "error": "Ground-truth mapping error", "original_sample": sample, "token_usage": token_usage}

    # 不传入 scenario/domain/culture
    def build_user_text(include_audio_caption: bool) -> str:
        user_text = f"Text: {inp.get('text', '')}\n"
        if include_audio_caption and inp.get('audio_caption'):
            user_text += f"Audio Caption: {inp['audio_caption']}\n"

        user_text += "\n\nSubject slot mapping:\n" + "\n".join([f"{k} = {v}" for k, v in subject_slots.items()])
        user_text += "\n\nTarget slot mapping:\n" + "\n".join([f"{k} = {v}" for k, v in target_slots.items()])
        user_text += "\n\nCRITICAL REMINDER: For 'subject' and 'target', DO NOT copy the text descriptions. You MUST output the exact keys (e.g., 'subject0', 'target2')."
        user_text += "\n\nProvide the 4-field JSON response."
        return user_text

    media_path = resolve_media_path(sample_id, inp)
    is_video = False
    media_blocks_all = []
    try:
        if not media_path:
            raise FileNotFoundError("missing media path in sample input")
        is_video = str(media_path).lower().endswith(".mp4")
        if is_video:
            frames = extract_video_frames(
                media_path,
                Path("./_temp_eval_frames"),
                sample_id,
                RANDOM_SEED,
                VIDEO_NUM_FRAMES,
            )
            if not frames:
                raise RuntimeError("video frame extraction produced 0 frames")
            if len(frames) < VIDEO_NUM_FRAMES:
                print(f"[FrameWarning] id={sample_id} extracted_frames={len(frames)}/{VIDEO_NUM_FRAMES}", flush=True)
            for f in frames:
                media_blocks_all.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(f)}})
        else:
            if not Path(media_path).exists():
                raise FileNotFoundError(media_path)
            media_blocks_all.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(media_path)}})
    except Exception as e:
        return {"id": sample_id, "error": f"Media error: {e}", "token_usage": token_usage}

    prediction = None
    api_error = None
    raw_pred_mech = ""
    raw_pred_label = ""
    raw_pred_sub = ""
    raw_pred_tgt = ""

    sit_norm = normalize_text(given_scenario)
    if sit_norm == "affection":
        system_prompt = SYSTEM_PROMPT_AFFECTION
    elif sit_norm == "attitude":
        system_prompt = SYSTEM_PROMPT_ATTITUDE
    elif sit_norm == "intent":
        system_prompt = SYSTEM_PROMPT_INTENT
    else:
        return {"id": sample_id, "error": f"Unknown scenario: {given_scenario}", "token_usage": token_usage}

    fallback_plan = []
    seen_modes = set()

    def add_mode(include_audio_caption: bool, mode_name: str):
        key = (include_audio_caption,)
        if key in seen_modes:
            return
        seen_modes.add(key)
        fallback_plan.append(
            {
                "include_audio": include_audio_caption,
                "name": mode_name,
            }
        )

    has_audio_caption = bool(inp.get("audio_caption"))
    add_mode(True, "default")
    if has_audio_caption:
        add_mode(False, "drop_audio_caption")

    for mode_idx, mode in enumerate(fallback_plan):
        include_audio = mode["include_audio"]
        user_text = build_user_text(include_audio)
        content_blocks = [{"type": "text", "text": user_text}] + media_blocks_all

        context_overflow_hit = False
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content_blocks}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                one_usage = extract_usage_from_response(response)
                token_usage["prompt_tokens"] += one_usage["prompt_tokens"]
                token_usage["completion_tokens"] += one_usage["completion_tokens"]
                token_usage["total_tokens"] += one_usage["total_tokens"]
                token_usage["cached_tokens"] += one_usage["cached_tokens"]
                if response.choices[0].message.content:
                    prediction = json.loads(response.choices[0].message.content)
                    # 记录原始生成的错误文本用于查验
                    raw_pred_mech = prediction.get("mechanism", "")
                    raw_pred_label = prediction.get("label", "")
                    raw_pred_sub = prediction.get("subject", "")
                    raw_pred_tgt = prediction.get("target", "")

                    api_error = None
                    break
            except Exception as e:
                api_error = str(e)
                if is_context_overflow_error(api_error):
                    context_overflow_hit = True
                    break
                time.sleep(2)

        if prediction is not None:
            break

        if context_overflow_hit and mode_idx < len(fallback_plan) - 1:
            print(
                f"[ContextFallback] id={sample_id} mode={mode['name']} "
                f"error={api_error[:180]} -> next_mode={fallback_plan[mode_idx + 1]['name']}",
                flush=True,
            )
            continue

    if api_error: return {"id": sample_id, "error": api_error, "token_usage": token_usage}

    # 错误分析：基于给定的 scenario 校验机制和标签是否合法 (对齐 Qwen 5. 部分逻辑)
    sit_norm = normalize_text(given_scenario)
    mech_norm = normalize_text(raw_pred_mech).replace("_", " ")
    label_norm = normalize_text(raw_pred_label).replace("_", " ")
    allowed_mechs = [m.replace("_", " ") for m in VALID_MECHANISMS.get(sit_norm, [])]
    # Compare labels case-insensitively to avoid false mismatches like Provoke vs provoke.
    allowed_labels = [normalize_text(l).replace("_", " ") for l in VALID_LABELS.get(sit_norm, [])]

    error_analysis = {
        "mech_mismatch": mech_norm not in allowed_mechs if allowed_mechs else True,
        "label_mismatch": label_norm not in allowed_labels if allowed_labels else True,
        "subject_format_error": normalize_slot(raw_pred_sub) not in {"subject0", "subject1", "subject2", "subject3"},
        "target_format_error": normalize_slot(raw_pred_tgt) not in {"target0", "target1", "target2", "target3"},
        "predicted_mechanism": raw_pred_mech, "predicted_label": raw_pred_label,
        "predicted_subject": raw_pred_sub, "predicted_target": raw_pred_tgt
    }

    # 判定判定
    fields = ["mechanism", "label", "subject", "target"]
    matches = {}
    is_strict = True
    for f in fields:
        if f in ["subject", "target"]:
            m = (normalize_slot(prediction.get(f, "")) == (true_subject_slot if f == "subject" else true_target_slot))
        else:
            m = (normalize_text(prediction.get(f, "")) == normalize_text(out.get(f, "")))
        matches[f] = m
        if not m: is_strict = False

    return {
        "id": sample_id,
        "ground_truth": {**{f: out.get(f) for f in fields}, "subject": true_subject_slot, "target": true_target_slot},
        "prediction": prediction, "matches": matches, "strict_match": is_strict, "error_analysis": error_analysis,
        "meta_scenario": given_scenario, "meta_domain": out.get("domain"), "meta_culture": out.get("culture"),
        "token_usage": token_usage,
        "slot_maps": {"subject_slots": subject_slots, "target_slots": target_slots,
                      "true_subject_slot": true_subject_slot, "true_target_slot": true_target_slot}
    }


# ==========================================
# 4. 指标计算与可视化 (对齐 Qwen 标准化预测分 + Strict All柱子)
# ==========================================
def calculate_metrics_for_subset(results_list):
    if not results_list: return {"Count": 0}
    metrics = {"Count": len(results_list)}
    metrics["Strict_All4_Acc"] = accuracy_score([1] * len(results_list),
                                                [1 if r['strict_match'] else 0 for r in results_list])

    fields = ["mechanism", "label", "subject", "target"]
    for f in fields:
        y_true = [normalize_text(r['ground_truth'].get(f, "")) for r in results_list]
        y_pred = []
        for r in results_list:
            p = normalize_text(r['prediction'].get(f, ""))
            if f in ["subject", "target"] and p not in {"subject0", "subject1", "subject2", "subject3", "target0",
                                                        "target1", "target2", "target3"}:
                y_pred.append("invalid_format")
            else:
                y_pred.append(p)
        metrics[f"{f}_Accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{f}_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return metrics


def plot_metrics(report, output_path, successful_count, model_name):
    overall = report[f"1. Overall ({successful_count} Successful Samples)"]
    # 仅绘制模型真实预测的 4 个字段，以及 Strict Match (对齐 Qwen 逻辑)
    plot_fields = ["mechanism", "label", "subject", "target", "strict_all"]
    accs = [overall.get(f"{f}_Accuracy", overall.get('Strict_All4_Acc') if f == "strict_all" else 0) for f in
            plot_fields]
    f1s = [overall.get(f"{f}_F1", 0) for f in plot_fields]

    x = np.arange(len(plot_fields))
    width = 0.35
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accs, width, label='Accuracy', color='#2b83ba', edgecolor='black', linewidth=0.7)
    ax.bar(x + width / 2, f1s, width, label='Macro F1', color='#d7191c', edgecolor='black', linewidth=0.7)
    ax.set_ylabel('Scores', fontsize=12, weight='bold')
    ax.set_title(f'Performance of {model_name} (N={successful_count})', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize().replace("_", " ") for f in plot_fields], rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 5. 主干逻辑 (增加第 5 部分错误报告)
# ==========================================
def main():
    with open(INPUT_JSON_PATH, 'r') as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        raw_items = loaded
    elif isinstance(loaded, dict) and isinstance(loaded.get("data"), list):
        raw_items = loaded["data"]
    else:
        raise ValueError(f"Unsupported JSON structure in INPUT_JSON_PATH: {INPUT_JSON_PATH}")
    combined = adapt_samples_to_legacy(raw_items)
    if not combined:
        raise ValueError(f"No valid samples loaded from INPUT_JSON_PATH: {INPUT_JSON_PATH}")
    random.seed(RANDOM_SEED)
    sampled = random.sample(combined, min(SAMPLE_SIZE, len(combined)))
    client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
    print(
        f"Config: model={MODEL_NAME} base_url={VLLM_BASE_URL} sample_size={len(sampled)} "
        f"seed={RANDOM_SEED} workers={MAX_WORKERS} video_frames={VIDEO_NUM_FRAMES} "
        f"max_output_tokens={MAX_OUTPUT_TOKENS} "
        f"ffmpeg_available={FFMPEG_AVAILABLE} local_media_index={len(LOCAL_MEDIA_BY_ID)}"
    , flush=True)

    results, failed = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, s, client): s for s in sampled}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            if "error" not in res:
                results.append(res)
            else:
                failed.append(res)
                print(f"[Failed] id={res.get('id')} error={res['error']}", flush=True)
            if i % 10 == 0:
                print(f"Progress: {i}/{len(sampled)} success={len(results)} failed={len(failed)}", flush=True)

    with open(OUTPUT_DETAILED_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_FAILED_FILE, 'w') as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
    token_usage_rows = []
    for r in results:
        usage = r.get("token_usage", {})
        token_usage_rows.append({
            "id": r.get("id"),
            "status": "success",
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "cached_tokens": int(usage.get("cached_tokens", 0)),
        })
    for r in failed:
        usage = r.get("token_usage", {})
        token_usage_rows.append({
            "id": r.get("id"),
            "status": "failed",
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "cached_tokens": int(usage.get("cached_tokens", 0)),
            "error": r.get("error"),
        })
    with open(OUTPUT_TOKEN_USAGE_FILE, 'w') as f:
        json.dump(token_usage_rows, f, indent=2, ensure_ascii=False)

    if not results: return
    report = {"1. Overall": calculate_metrics_for_subset(results)}
    for group_name, key in [("2. By Scenario", "meta_scenario"), ("3. By Domain", "meta_domain"),
                            ("4. By Culture", "meta_culture")]:
        report[group_name] = {val: calculate_metrics_for_subset([r for r in results if r[key] == val]) for val in
                              globals()[group_name.split()[-1].upper() + 'S']}

    # 第 5 部分：错误分析报告 (对齐 Qwen)
    err_rep = {"mechanism_mismatches": {"count": 0, "samples": []}, "label_mismatches": {"count": 0, "samples": []},
               "format_errors": {"count": 0, "samples": []}}
    for r in results:
        ea = r["error_analysis"]
        if ea["mech_mismatch"]:
            err_rep["mechanism_mismatches"]["count"] += 1
            err_rep["mechanism_mismatches"]["samples"].append(
                {"id": r["id"], "given_sit": r["meta_scenario"], "pred_mech": ea["predicted_mechanism"]})
        if ea["label_mismatch"]:
            err_rep["label_mismatches"]["count"] += 1
            err_rep["label_mismatches"]["samples"].append(
                {"id": r["id"], "given_sit": r["meta_scenario"], "pred_label": ea["predicted_label"]})
        if ea["subject_format_error"] or ea["target_format_error"]:
            err_rep["format_errors"]["count"] += 1
            err_rep["format_errors"]["samples"].append(
                {"id": r["id"], "pred_sub": ea["predicted_subject"], "pred_tgt": ea["predicted_target"]})

    overall_key = f"1. Overall ({len(results)} Successful Samples)"
    overall_block = report.pop("1. Overall")
    report = {
        overall_key: overall_block,
        **report,
        "5. Error Analysis": err_rep,
    }

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    plot_metrics(report, OUTPUT_PLOT_FILE, len(results), MODEL_NAME)
    print(f"Done! Strict Acc: {report[f'1. Overall ({len(results)} Successful Samples)']['Strict_All4_Acc']:.2%}")
    print(f"Saved token usage file: {OUTPUT_TOKEN_USAGE_FILE}")


if __name__ == "__main__":
    main()
