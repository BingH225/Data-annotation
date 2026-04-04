import os
import json
import time
import random
import base64
import hashlib
import subprocess
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent_framework import IntentSotaConfig, IntentSotaPipeline, SubjectTargetAgentPipeline

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ==========================================
# 1. 閰嶇疆鍖哄煙
# ==========================================
# OpenAI-compatible config
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()
if not SILICONFLOW_API_KEY:
    SILICONFLOW_API_KEY = os.environ.get("VLLM_API_KEY", "").strip()
if not SILICONFLOW_API_KEY:
    SILICONFLOW_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not SILICONFLOW_API_KEY:
    SILICONFLOW_API_KEY = "EMPTY"

SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "").strip()
if not SILICONFLOW_BASE_URL:
    SILICONFLOW_BASE_URL = os.environ.get("VLLM_BASE_URL", "").strip()
if not SILICONFLOW_BASE_URL:
    SILICONFLOW_BASE_URL = "http://127.0.0.1:8000/v1"

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_NAME = os.environ.get("EVAL_MODEL_NAME", "").strip()
if not MODEL_NAME:
    MODEL_NAME = os.environ.get("MODEL_NAME", "/hpctmp/e1561245/qwen_project/Qwen3-VL-32B-Thinking").strip()

DEFAULT_INPUT_JSON_PATH = SCRIPT_DIR / "data" / "eval300_affection100_from_baseline300.json"
INPUT_JSON_PATH = os.environ.get("INPUT_JSON_PATH", str(DEFAULT_INPUT_JSON_PATH)).strip()

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "0"))  # 0 means full data (no sampling)
RANDOM_SEED = 42
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
MAX_IMAGES_PER_PROMPT = int(os.environ.get("MAX_IMAGES_PER_PROMPT", "8"))
SUBJECT_TARGET_PIPELINE_STRATEGY = os.environ.get("SUBJECT_TARGET_PIPELINE_STRATEGY", "rjg_role_fusion").strip() or "rjg_role_fusion"
try:
    SUBJECT_TARGET_ANCHOR_MIN_CONF = float(os.environ.get("SUBJECT_TARGET_ANCHOR_MIN_CONF", "0.55"))
except Exception:
    SUBJECT_TARGET_ANCHOR_MIN_CONF = 0.55
try:
    SUBJECT_TARGET_ROLE_PROBE_MIN_CONF = float(os.environ.get("SUBJECT_TARGET_ROLE_PROBE_MIN_CONF", "0.55"))
except Exception:
    SUBJECT_TARGET_ROLE_PROBE_MIN_CONF = 0.55
SUBJECT_TARGET_ROLE_PROBE_ENABLED = str(os.environ.get("SUBJECT_TARGET_ROLE_PROBE_ENABLED", "1")).strip().lower() not in {"0", "false", "no"}
SUBJECT_TARGET_SHUFFLE_OPTIONS = str(os.environ.get("SUBJECT_TARGET_SHUFFLE_OPTIONS", "0")).strip().lower() in {"1", "true", "yes"}
SUBJECT_TARGET_BASELINE_ROLE_PROBE_ENABLED = str(
    os.environ.get("SUBJECT_TARGET_BASELINE_ROLE_PROBE_ENABLED", "1")
).strip().lower() not in {"0", "false", "no"}
try:
    SUBJECT_TARGET_BASELINE_ROLE_PROBE_MIN_CONF = float(
        os.environ.get("SUBJECT_TARGET_BASELINE_ROLE_PROBE_MIN_CONF", "0.60")
    )
except Exception:
    SUBJECT_TARGET_BASELINE_ROLE_PROBE_MIN_CONF = 0.60

INTENT_SOTA_STAGE_A_VOTES = int(os.environ.get("INTENT_SOTA_STAGE_A_VOTES", "5"))
INTENT_SOTA_STAGE_A_TOP_K = int(os.environ.get("INTENT_SOTA_STAGE_A_TOP_K", "4"))
try:
    INTENT_SOTA_STAGE_A_ALT_WEIGHT = float(os.environ.get("INTENT_SOTA_STAGE_A_ALT_WEIGHT", "0.35"))
except Exception:
    INTENT_SOTA_STAGE_A_ALT_WEIGHT = 0.35
INTENT_SOTA_STAGE_C_VOTES = int(os.environ.get("INTENT_SOTA_STAGE_C_VOTES", "5"))
try:
    INTENT_SOTA_LABEL_SIGNAL_CONF_THRESHOLD = float(os.environ.get("INTENT_SOTA_LABEL_SIGNAL_CONF_THRESHOLD", "0.58"))
except Exception:
    INTENT_SOTA_LABEL_SIGNAL_CONF_THRESHOLD = 0.58
try:
    INTENT_SOTA_HARD_REFINE_TOP_CONF_THRESHOLD = float(os.environ.get("INTENT_SOTA_HARD_REFINE_TOP_CONF_THRESHOLD", "0.72"))
except Exception:
    INTENT_SOTA_HARD_REFINE_TOP_CONF_THRESHOLD = 0.72
INTENT_SOTA_DISABLE_PRIOR_ROUTER = str(os.environ.get("INTENT_SOTA_DISABLE_PRIOR_ROUTER", "0")).strip().lower() in {"1", "true", "yes"}
INTENT_SOTA_PRIOR_BASE_FILE = os.environ.get("INTENT_SOTA_PRIOR_BASE_FILE", "").strip()
INTENT_SOTA_PRIOR_COT_FILE = os.environ.get("INTENT_SOTA_PRIOR_COT_FILE", "").strip()
try:
    INTENT_SOTA_PRIOR_MECH_THRESHOLD = float(os.environ.get("INTENT_SOTA_PRIOR_MECH_THRESHOLD", "0.66"))
except Exception:
    INTENT_SOTA_PRIOR_MECH_THRESHOLD = 0.66
try:
    INTENT_SOTA_PRIOR_BASE_ONLY_MECH_THRESHOLD = float(os.environ.get("INTENT_SOTA_PRIOR_BASE_ONLY_MECH_THRESHOLD", "0.76"))
except Exception:
    INTENT_SOTA_PRIOR_BASE_ONLY_MECH_THRESHOLD = 0.76
try:
    INTENT_SOTA_PRIOR_LABEL_THRESHOLD = float(os.environ.get("INTENT_SOTA_PRIOR_LABEL_THRESHOLD", "0.68"))
except Exception:
    INTENT_SOTA_PRIOR_LABEL_THRESHOLD = 0.68
SCENARIOS = ["affection", "attitude", "intent"]
DOMAINS = ["Online & Social Media", "Public & Service", "Workplace", "Intimate Relationships", "Family Conversations",
           "Friend Group", "Education & Campus", "Friendship Interactions"]
CULTURES = ["General Culture", "Arab Culture", "American Culture", "Muslim Culture", "African American Culture",
            "Jewish Culture", "Indian Culture", "East Asian Culture"]

# ==========================================
# 鍔ㄦ€佸垱寤鸿緭鍑烘枃浠跺す (瀵归綈 Qwen 鍛藉悕椋庢牸)
# ==========================================
SAFE_MODEL_NAME_FOR_PATH = MODEL_NAME.replace("/", "_").replace("\\", "_").replace(":", "-")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "baseline100_v2"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)).strip())
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DETAILED_FILE = OUTPUT_DIR / "evaluation_predictions_detailed.json"
OUTPUT_METRICS_FILE = OUTPUT_DIR / "evaluation_metrics_report.json"
OUTPUT_FAILED_FILE = OUTPUT_DIR / "evaluation_failures.json"
OUTPUT_TOKEN_USAGE_FILE = OUTPUT_DIR / "evaluation_token_usage_by_id.json"
OUTPUT_PLOT_FILE = OUTPUT_DIR / "evaluation_metrics_plot.png"
OUTPUT_FORMULA_METRICS_FILE = OUTPUT_DIR / "evaluation_formula_metrics.json"
OUTPUT_FORMULA_TABLE_FILE = OUTPUT_DIR / "evaluation_formula_metrics_table_ready.json"

# 鍏ㄥ眬鏈夋晥鍒楄〃锛岀敤浜庨敊璇粺璁℃牎楠?(瀵归綈 Qwen 璇嶈〃)
VALID_MECHANISMS = {
    "affection": ["multimodal incongruity", "figurative semantics", "affective deception", "socio_cultural dependency"],
    "intent": ["prosocial deception", "malicious manipulation", "expressive aggression", "benevolent provocation"],
    "attitude": ["dominant affiliation", "dominant detachment", "protective distancing", "submissive alignment"]
}
VALID_LABELS = {
    "affection": ["happy", "sad", "disgusted", "angry", "fearful", "bad"],
    "attitude": ["supportive", "appreciative", "sympathetic", "neutral",
                 "indifferent", "concerned", "skeptical", "dismissive", "disapproving", "contemptuous", "hostile"],
    "intent": ["mitigate", "intimidate", "alienate", "mock", "denounce", "provoke", "dominate", "condemn"]
}

SUBJECT_TARGET_GROUNDING_STEPS = """
Subject/Target grounding steps (perform internally before the final JSON):
Step 1 - Explicit perception (structured decomposition):
- Internally decompose the multimodal input into this structure:
  {
    "text_components": {"subject": "", "object": "", "predicate": "", "attribute": "", "adverbial": ""},
    "image_action": {"subject": "", "background": "", "behavior": "", "action": ""},
    "audio_caption": {"subject": "", "object": "", "predicate": "", "attribute": "", "adverbial": ""}
  }
- Fill unavailable fields as empty strings.
Step 2 - Social context knowledge graph (Subject Relations):
- Build an internal lightweight graph from Step 1 entities.
- Graph requirement: nodes are entities/roles; directed edges describe who addresses/evaluates/acts on whom and include contextual cues (power, intimacy, history/context hints when available).
- Use this graph to identify one main subject slot and one main target slot.
Do NOT output these steps or any reasoning. Output only the final 4-field JSON.
"""

SUBJECT_TARGET_AGENT = SubjectTargetAgentPipeline(
    strategy=SUBJECT_TARGET_PIPELINE_STRATEGY,
    anchor_min_conf=SUBJECT_TARGET_ANCHOR_MIN_CONF,
    role_probe_min_conf=SUBJECT_TARGET_ROLE_PROBE_MIN_CONF,
)

INTENT_SOTA_AGENT: Optional[IntentSotaPipeline] = None


def build_intent_sota_agent(client: OpenAI) -> IntentSotaPipeline:
    cfg = IntentSotaConfig(
        stage_a_votes=max(1, int(INTENT_SOTA_STAGE_A_VOTES)),
        stage_a_top_k=max(1, int(INTENT_SOTA_STAGE_A_TOP_K)),
        stage_a_alt_weight=float(INTENT_SOTA_STAGE_A_ALT_WEIGHT),
        stage_c_votes=max(1, int(INTENT_SOTA_STAGE_C_VOTES)),
        label_signal_conf_threshold=float(INTENT_SOTA_LABEL_SIGNAL_CONF_THRESHOLD),
        hard_refine_top_conf_threshold=float(INTENT_SOTA_HARD_REFINE_TOP_CONF_THRESHOLD),
        disable_prior_router=bool(INTENT_SOTA_DISABLE_PRIOR_ROUTER),
        prior_base_file=str(INTENT_SOTA_PRIOR_BASE_FILE or ""),
        prior_cot_file=str(INTENT_SOTA_PRIOR_COT_FILE or ""),
        prior_mech_threshold=float(INTENT_SOTA_PRIOR_MECH_THRESHOLD),
        prior_base_only_mech_threshold=float(INTENT_SOTA_PRIOR_BASE_ONLY_MECH_THRESHOLD),
        prior_label_threshold=float(INTENT_SOTA_PRIOR_LABEL_THRESHOLD),
    )
    return IntentSotaPipeline(client=client, model_name=MODEL_NAME, max_retries=MAX_RETRIES, config=cfg)


# ==========================================
# System Prompt (鐩存帴鎷疯礉鑷綘鐨?qwen_eva_sit_dc.py)
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
- "multimodal incongruity": Implicit affection arises from polarity conflict or mutual exclusion between modalities (text vs image, or image-context vs text), such that the literal meaning is negated or reframed by the other modality. Key signature: "What is said" and "what is shown" cannot both be true in the same frame -> the affective state is inferred from the conflict.
- "figurative semantics": The affective state is conveyed via source->target conceptual mapping rather than direct emotion words or standard displays. Key signature: The sample "talks about X" but means an affective state through metaphor, symbol, hyperbole, understatement, or poetic imagery.
- "affective deception": The affective state is deliberately masked (performed calmness/harshness), but involuntary cues "leak" the underlying affect. Key signature: "Displayed affect" != "true affective state" inferred from leakage.
- "socio_cultural context dependency": The affective state can only be interpreted correctly using external world knowledge (memes, events, cultural codes, relationship norms). Key signature: The pair is semantically opaque without a shared cultural reference that encodes affect indirectly.

2. "label": Classify the core emotion into exactly ONE of the primary labels below. You must first analyze the underlying mechanism of the scenario, then use the provided Level-2 Sub-labels and Boundary Rules to map your analysis to the correct primary label:

- happy
  * Level-2 Sub-labels: Playful, Content, Interested, Proud, Accepted, Powerful, Peaceful, Trusting, Optimistic.
  * Core Mechanism: A positive state involving satisfaction, joy, or confidence, usually accompanied by positive evaluation of the current scenario (e.g., feeling accepted or hopeful).
  * Boundary Rule: If the expression is primarily factual with extremely weak emotion and lacks clear positive cues, classify as 'Bad' as the low-intensity fallback required by this label set. If the core mechanism is gratitude/thankfulness for someone's help, it strictly falls under 'Happy'.
- sad
  * Level-2 Sub-labels: Lonely, Vulnerable, Despair, Guilty, Depressed, Hurt.
  * Core Mechanism: A negative state associated with loss, helplessness, or relationship setbacks, often leading to withdrawal or lack of motivation.
  * Boundary Rule: If the focus is on self-blame/compensation for a mistake, map to the 'Guilty' sub-label. If the focus is relational pain caused by others, map to the 'Hurt' sub-label. 
- disgusted
  * Level-2 Sub-labels: Repelled, Awful, Disappointed, Disapproving.
  * Core Mechanism: A strong aversion or rejection toward physical stimuli (smell/food) or social/moral stimuli (immorality, hypocrisy, offense).
  * Boundary Rule: The core tendency is "AWAY FROM" (want to avoid/reject/deny). If the emotion primarily involves blame, confrontation, or emphasizing "the other party is wrong" rather than mere aversion, it leans towards 'Angry'.
- angry
  * Level-2 Sub-labels: Let down, Humiliated, Bitter, Mad, Aggressive, Frustrated, Distant, Critical.
  * Core Mechanism: Triggered by being offended, hindered, or treated unfairly, leading to hostility, frustration, and an intention to control, fight back, or argue.
  * Boundary Rule: The core tendency is "AGAINST". If the expression is merely cold or "I don't care" (without confrontation), it leans towards 'Bad'. If it's primarily aversion without the urge to fight back, it leans towards 'Disgusted'.
- fearful
  * Level-2 Sub-labels: Scared, Anxious, Insecure, Weak, Rejected, Threatened.
  * Core Mechanism: Centered around threat and insecurity, expecting that "something bad might happen / I might get hurt," leading to a tendency to avoid, seek protection, or increase vigilance.
  * Boundary Rule: Includes continuous worry caused by long-term uncertainty ('Anxious'). If it's just a brief reaction to unexpected factual info without a core sense of threat, map it to 'Bad' based on context.
- bad
  * Level-2 Sub-labels: Bored, Busy, Stressed, Tired.
  * Core Mechanism: A generalized negative state for scenarios that do not fit the specific intensity or clear triggers of Sad, Angry, Disgusted, or Fearful. It includes feeling uncomfortable, exhausted, apathetic, or experiencing cognitive dissonance.
  * Boundary Rule: Use this category for low-intensity negative states or mixed/cynical contexts. For instance, implicit scenarios involving "dark humor" (鍦扮嫳绗戣瘽) should be classified here. The mechanism of dark humor relies on taboo, tragedy, or moral discomfort, which aligns with this generalized uncomfortable/apathetic state, rather than genuine positive joy (Happy) or targeted hostility (Angry).

3. "subject": You MUST select EXACTLY ONE string from this strict list: ["subject0", "subject1", "subject2", "subject3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

4. "target": You MUST select EXACTLY ONE string from this strict list: ["target0", "target1", "target2", "target3"].
WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.

# Output Format Requirements
Output ONLY a strict JSON object with EXACTLY 4 keys. You MUST adhere to the exact values requested. Use this exact schema:
{
  "mechanism": "<multimodal incongruity | figurative semantics | affective deception | socio_cultural context dependency>",
  "label": "<happy | sad | disgusted | angry | fearful | bad>",
  "subject": "<subject0 | subject1 | subject2 | subject3>",
  "target": "<target0 | target1 | target2 | target3>"
}
Do NOT write markdown blocks (no json).
Do NOT include any explanations, rationales, or chain-of-thought reasoning.
"""



SYSTEM_PROMPT_ATTITUDE = SYSTEM_PROMPT_BASE + """
You must analyze the sample and determine the correct values for 4 specific keys. 

1. "mechanism": Choose exactly ONE from the list below:
- "dominant affiliation": Surface friendliness or acceptance is used to assert a "stronger to weaker" superiority; closeness is granted from above. Key signatures include: Talking Down (treating the target like a child to establish superiority), Patronizing Praise (praising while implicitly lowering the standard for the target), or Benevolent Control (depriving the target's autonomy under the guise of "for your own good").
- "dominant detachment": Establishing high status while cutting off emotional connection. Key signatures include: Invalidation (standing on a moral/intellectual high ground to deny the target's value or feelings), Character Blaming (attributing the target's plight to their own moral/personality flaws; "they asked for it"), or Rationalizing Over (using cold logic to dismiss the target's perspective as unworthy of serious attention).
- "protective distancing": A lower-power or vulnerable stance avoiding direct confrontation via non-commitment or psychological isolation. Key signatures include: Keeping It Open (suspending commitment/action without explicitly agreeing or disagreeing), Emotional Withdrawal (receiving information but noticeably dropping emotional engagement), or Skeptical Distance (appearing open but implicitly refusing to accept the target's premises).
- "submissive alignment": Proactively lowering one's own status to seek safety, attachment, or protection. Key signatures include: Self-Diminishment (belittling oneself to gain acceptance), Over-Accommodation (sacrificing one's own stance to avoid conflict/rejection), or Leaning On (showing weakness to signal a need for the target's protection or decision-making).

2. "label": Classify the implicit attitude into EXACTLY ONE of the labels below based on your mechanism analysis. Use the Anchor Examples contextually and strictly apply the Boundary Rules:

- "supportive"
  * Core Definition: Explicitly defends the target or affirms their legitimacy and value.
  * Anchor Examples (Non-exhaustive): e.g., making excuses for their failure, protecting their reputation, taking their side.
  * Boundary Rule: The focus is on active alignment/defense. If it only evaluates output positively without defending or taking a side, map to 'Appreciative'.
- "appreciative"
  * Core Definition: Positive evaluation of the target's abilities, qualities, or achievements.
  * Anchor Examples (Non-exhaustive): e.g., praising effort, complimenting results or design.
  * Boundary Rule: The focus is on merit evaluation. It differs from 'Supportive' as it doesn't necessarily involve defending the target against adversity.
- "sympathetic"
  * Core Definition: Empathy and understanding for the target's unfavorable scenario.
  * Anchor Examples (Non-exhaustive): e.g., emphasizing harsh environments, bad luck, or high difficulty to comfort and downplay responsibility.
  * Boundary Rule: The focus is on shared suffering/excusing circumstances. If the subject actively defends the target's actions as fully correct, it leans towards 'Supportive'.
- "neutral"
  * Core Definition: No clear value judgment or affective stance.
  * Anchor Examples (Non-exhaustive): e.g., objective statements of facts, news-like reports.
  * Boundary Rule: Use ONLY if the expression is purely factual. If there is a deliberate, cynical withdrawal of care, use 'Indifferent'.
- "indifferent"
  * Core Definition: Explicitly conveying a lack of care or engagement regarding the target.
  * Anchor Examples (Non-exhaustive): e.g., "whatever," "doesn't matter," showing apathy.
  * Boundary Rule: The focus is on apathy. If the subject actively downplays the target's importance to shut them down, use 'Dismissive'.
- "disapproving"
  * Core Definition: Negative evaluation of a specific behavior, decision, or choice, while still treating the target as an equal whose overall value is intact.
  * Anchor Examples (Non-exhaustive): e.g., being critical, pointing out flaws, reproachful about a specific issue.
  * Boundary Rule: The focus MUST be on the action/issue, not the person. If the evaluation attacks the target's inherent worth or is delivered from a position of superiority, classify as 'Contemptuous'.
- "skeptical"
  * Core Definition: Holding reservations about the target's authenticity, ability, claims, or motives.
  * Anchor Examples (Non-exhaustive): e.g., doubting assumptions, hinting at unreliability, guarded belief.
  * Boundary Rule: The focus is on doubt. If the subject outright denies the validity or worth of the target without consideration, classify as 'Dismissive'.
- "concerned"
  * Core Definition: Belief that the target might bring risk, harm, or negative consequences.
  * Anchor Examples (Non-exhaustive): e.g., reminding of risks, using a cautious tone, expressing worry.
  * Boundary Rule: The focus is on potential future negative outcomes. If the subject expresses empathy for an already occurred negative outcome, use 'Sympathetic'.
- "dismissive"
  * Core Definition: Denying the importance, reasonableness, or discussion value of the target or their standpoint.
  * Anchor Examples (Non-exhaustive): e.g., brushing off, ignoring, treating the input as trivial, using irony to downplay.
  * Boundary Rule: The core action is ignoring or shutting down. It differs from 'Disapproving' because it doesn't seriously evaluate the action; it differs from 'Contemptuous' because it doesn't necessarily attack the target's dignity.
- "contemptuous"
  * Core Definition: Viewing the target as inherently inferior, worthless, or beneath respect.
  * Anchor Examples (Non-exhaustive): e.g., acting disdainful, mocking, arrogant, patronizing, condescension.
  * Boundary Rule: The focus is on degrading the person. If the subject is merely denying the importance of a topic without personal attacks, map to 'Dismissive'.
- "hostile"
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
  "mechanism": "<dominant affiliation | dominant detachment | protective distancing | submissive alignment>",
  "label": "<supportive | appreciative | sympathetic | neutral | indifferent | disapproving | skeptical | concerned | dismissive | contemptuous | hostile>",
  "subject": "<subject0 | subject1 | subject2 | subject3>",
  "target": "<target0 | target1 | target2 | target3>"
}
Do NOT write markdown blocks (no json).
Do NOT include any explanations, rationales, or chain-of-thought reasoning.
"""


SYSTEM_PROMPT_INTENT = SYSTEM_PROMPT_BASE + """
You must analyze the sample and determine the correct values for 4 specific keys. 

1. "mechanism": Choose exactly ONE from the list below:
- "prosocial deception": A benevolent concealment where the speaker prioritizes social harmony or face-saving over factual accuracy. Key signatures: "White lies" (masking true negative feelings like disgust or disappointment with surface positivity to protect the target's feelings or avoid direct rejection).
- "malicious manipulation": A strategic intent to exploit human vulnerabilities while packaging harm or control as help, vulnerability, or morality. Key signatures: "Killing with kindness" (over-praising or indulging to set the target up for a fall), "Playing the victim" (exaggerating one's own suffering to guilt-trip or frame others), or "Moral kidnapping" (using high moral standards to force compliance or stigmatize the target).
- "expressive aggression": Indirect or veiled hostility used to attack, mock, or control the target while maintaining a facade. Key signatures: "Veiled abuse" (publicly mocking or insulting under the guise of giving advice/gifts) or "Implicit threats" (using overly specific observations or fake care to instill psychological pressure and fear).
- "benevolent provocation": A strategic disguise used to trigger a desired action or reveal the truth. Key signatures: "Goading" or "Reverse psychology" (creating a fake scenario, pretending ignorance, or challenging the target to force them to expose their true stance or confess).

2. "label": Classify the core intent into EXACTLY ONE of the labels below based on your mechanism analysis. Use the Anchor Examples contextually and strictly apply the Boundary Rules:

- "mitigate"
  * Core Definition: Intent to de-escalate tension, save face, and maintain a cooperative relationship.
  * Anchor Examples (Non-exhaustive): e.g., offering a compromise, smoothing over an awkward scenario, telling a prosocial "white lie".
  * Boundary Rule: The primary goal MUST be reducing friction or protecting harmony. If the subject uses fake empathy merely to set up an attack later, classify as 'Provoke' or 'Dominate'.
- "intimidate"
  * Core Definition: Intent to coerce the target through fear, implied consequences, or threats.
  * Anchor Examples (Non-exhaustive): e.g., implicit threats, leveraging specific knowledge to instill psychological pressure, forcing compliance.
  * Boundary Rule: The focus is on instilling fear for control. If the aggression is purely for amusement without the goal of coercing behavior, classify as 'Mock'.
- "alienate"
  * Core Definition: Intent to isolate the target by stripping them of equal status and reducing them to an "outsider" or negative stereotype.
  * Anchor Examples (Non-exhaustive): e.g., othering ("us vs. them"), dehumanizing, group-based exclusion.
  * Boundary Rule: The focus is on identity/group exclusion. If the public isolation is based on a specific moral failing rather than identity/stereotyping, map to 'Denounce'.
- "mock"
  * Core Definition: Intent to entertain oneself or others at the target's expense by making them look foolish.
  * Anchor Examples (Non-exhaustive): e.g., treating flaws as a joke, humiliation for amusement, teasing.
  * Boundary Rule: The primary goal is amusement. If the humor/mockery is used as a veil for a severe attack or baiting to trigger anger, classify as 'Provoke'. If the goal is strictly asserting power, map to 'Dominate'.
- "denounce"
  * Core Definition: Intent to socially punish the target by exposing their actions to an audience and rallying social pressure.
  * Anchor Examples (Non-exhaustive): e.g., public shaming, canceling, rallying a crowd against the target.
  * Boundary Rule: MUST involve a public or multi-party audience dimension. If the judgment is purely one-on-one based on rules/ethics without rallying an audience, classify as 'Condemn'.
- "provoke"
  * Core Definition: Intent to bait, anger, or attack the target while maintaining plausible deniability, often using disguised hostility.
  * Anchor Examples (Non-exhaustive): e.g., using meme-based mockery, heavy irony, reverse psychology, goading to force a reaction.
  * Boundary Rule: The focus is on triggering a reaction or attacking from behind a shield of ambiguity. If the hostility is direct and aims to establish a clear hierarchy, map to 'Dominate'.
- "dominate"
  * Core Definition: Intent to establish power or intellectual asymmetry, forcing the target into a subordinate position.
  * Anchor Examples (Non-exhaustive): e.g., pulling rank, acting patronizingly, condescension, "killing with kindness".
  * Boundary Rule: The focus is on establishing a vertical hierarchy ("I am above you"). If the control is achieved primarily through fear rather than status assertion, map to 'Intimidate'.
- "condemn"
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
  "mechanism": "<prosocial deception | malicious manipulation | expressive aggression | benevolent provocation>",
  "label": "<mitigate | intimidate | alienate | mock | denounce | provoke | dominate | condemn>",
  "subject": "<subject0 | subject1 | subject2 | subject3>",
  "target": "<target0 | target1 | target2 | target3>"
}
Do NOT write markdown blocks (no json).
Do NOT include any explanations, rationales, or chain-of-thought reasoning.
"""


def _enable_entity_text_role_mode(prompt: str) -> str:
    text = str(prompt or "")
    text = text.replace(
        '3. "subject": You MUST select EXACTLY ONE string from this strict list: ["subject0", "subject1", "subject2", "subject3"].\n'
        "WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.\n\n"
        '4. "target": You MUST select EXACTLY ONE string from this strict list: ["target0", "target1", "target2", "target3"].\n'
        "WARNING: DO NOT output the descriptive text. You MUST output the placeholder ID only.",
        '3. "subject": You MUST select EXACTLY ONE entity text from the Subject options provided in user input.\n'
        "WARNING: Output the exact entity text only (not placeholder IDs like subject0).\n\n"
        '4. "target": You MUST select EXACTLY ONE entity text from the Target options provided in user input.\n'
        "WARNING: Output the exact entity text only (not placeholder IDs like target0).",
    )
    text = text.replace(
        '  "subject": "<subject0 | subject1 | subject2 | subject3>",\n'
        '  "target": "<target0 | target1 | target2 | target3>"',
        '  "subject": "<exact subject option text from Subject options>",\n'
        '  "target": "<exact target option text from Target options>"',
    )
    return text


SYSTEM_PROMPT_AFFECTION = _enable_entity_text_role_mode(SYSTEM_PROMPT_AFFECTION)
SYSTEM_PROMPT_ATTITUDE = _enable_entity_text_role_mode(SYSTEM_PROMPT_ATTITUDE)
SYSTEM_PROMPT_INTENT = _enable_entity_text_role_mode(SYSTEM_PROMPT_INTENT)

# ==========================================
# 2. 濯掍綋澶勭悊涓庡伐鍏峰嚱鏁?(淇濇寔鍍忕礌绾у師璨?
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


def build_frame_urls_from_folder(url_frame: str, sample_id: str, frame_count: int) -> list:
    """
    Build deterministic frame URLs from the pre-generated HF frame folder.
    Supports both /tree/main/ and /resolve/main/ folder links.
    """
    if not url_frame:
        return []
    base = str(url_frame).strip().rstrip("/")
    base = base.replace("/tree/main/", "/resolve/main/")
    if frame_count <= 0:
        return []
    return [f"{base}/{sample_id}_frame{i}.jpg" for i in range(1, frame_count + 1)]


def local_image_to_data_url(local_path: str) -> str:
    ext = Path(local_path).suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(Path(local_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def normalize_text(text):
    return str(text).strip().lower()


def normalize_slot(text):
    return str(text).strip().casefold()


def normalize_taxonomy_value(text):
    return " ".join(normalize_text(text).replace("_", " ").split())


def build_baseline_role_probe_prompts(
    *,
    scenario: str,
    text: str,
    audio_caption: str,
    subject_slots: Dict[str, str],
    target_slots: Dict[str, str],
) -> tuple[str, str]:
    system_prompt = (
        "You are a strict subject-target resolver for multimodal social annotation.\n"
        "Return JSON only with keys: subject, target, confidence, reason_short.\n"
        "Do not output mechanism or label.\n"
        "Select subject and target only from provided options."
    )
    user_prompt = (
        f"Scenario: {scenario}\n"
        f"Text: {text}\n"
        f"Audio Caption: {audio_caption}\n\n"
        "Subject options:\n"
        + "\n".join([f"- {v}" for v in subject_slots.values()])
        + "\n\nTarget options:\n"
        + "\n".join([f"- {v}" for v in target_slots.values()])
        + "\n\nRules:\n"
        + "1) Keep subject on the primary speaker/actor when text is speaker-centric.\n"
        + "2) Keep target on the most directly addressed/evaluated entity.\n"
        + "3) Use exact option text only; do not invent new entities.\n"
        + "4) Confidence must be in [0,1].\n\n"
        + "Return JSON only:\n"
        + "{\"subject\":\"<one subject option>\",\"target\":\"<one target option>\",\"confidence\":0.0,\"reason_short\":\"<=30 words\"}"
    )
    return system_prompt, user_prompt


def _pick_three_distractors(options_list, gt_value):
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
    Convert one item from exported_labels_new_format.json into the legacy
    structure expected by process_single_sample.
    """
    inp_new = sample.get("input", {})
    media = inp_new.get("media", {})
    options = sample.get("options", {})
    gt = sample.get("ground_truth", {})

    sample_id = sample.get("id") or inp_new.get("id") or inp_new.get("samples_id")
    scenario = inp_new.get("scenario", "")
    text = inp_new.get("text", "")
    audio_caption = media.get("audio_caption", "")
    url_frame = inp_new.get("url_frame", "")
    frame_count = inp_new.get("frame_count", 0)

    # Keep original media resolution logic: path/media_path_local + .mp4 check.
    video_path = media.get("video_path", "")
    image_path = media.get("image_path", "")
    media_path = video_path or image_path

    gt_subject = gt.get("subject", "")
    gt_target = gt.get("target", "")
    subject_distractors = _pick_three_distractors(options.get("subject", []), gt_subject)
    target_distractors = _pick_three_distractors(options.get("target", []), gt_target)

    legacy_input = {
        "id": sample_id,
        "text": text,
        "path": media_path
    }
    if video_path:
        legacy_input["media_path_local"] = video_path
    if audio_caption:
        legacy_input["audio_caption"] = audio_caption
    if url_frame:
        legacy_input["url_frame"] = url_frame
    try:
        frame_count_int = int(frame_count)
    except Exception:
        frame_count_int = 0
    if frame_count_int > 0:
        legacy_input["frame_count"] = frame_count_int

    legacy_output = {
        "scenario": scenario,
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

    if isinstance(data, dict):
        payload = data.get("data")
        if isinstance(payload, list):
            data = payload
        else:
            print("[LoadWarning] Unsupported top-level JSON format. Expected a list, or a dict with key 'data' containing a list.")
            return []

    if not isinstance(data, list):
        print("[LoadWarning] Unsupported top-level JSON type. Expected list-like samples.")
        return []

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
    for idx, sample in enumerate(data):
        try:
            converted.append(convert_new_format_to_legacy(sample))
        except Exception as e:
            sid = sample.get("id", f"idx_{idx}")
            print(f"[LoadWarning] skip sample {sid}: {e}")
    return converted


def accuracy_score_local(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def macro_f1_score_local(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0

    f1_list = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_list.append(f1)

    return sum(f1_list) / len(f1_list)


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


# ==========================================
# 3. 鏍稿績璇勪及娴佺▼ (瀵归綈 Qwen锛氭敞鍏?Scenario, Domain, Culture + 閿欒缁熻)
# ==========================================
def process_single_sample(sample, client):
    inp = sample['input']
    out = sample['output']
    sample_id = inp.get('id') or inp.get('samples_id')
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

    # 鑾峰彇鏍囧噯绛旀涓殑鍥哄畾瀛楁
    given_scenario = out.get("scenario", out.get("situation", ""))
    given_domain = out.get("domain", "")
    given_culture = out.get("culture", "")

    subjects_raw = [out.get('subject', ''), out.get('subject1', ''), out.get('subject2', ''), out.get('subject3', '')]
    targets_raw = [out.get('target', ''), out.get('target1', ''), out.get('target2', ''), out.get('target3', '')]
    subjects_raw = [s for s in subjects_raw if str(s).strip()]
    targets_raw = [t for t in targets_raw if str(t).strip()]

    if len(subjects_raw) != 4 or len(targets_raw) != 4:
        return {"id": sample_id, "error": f"Choices not exactly 4", "original_sample": sample, "token_usage": token_usage}

    if SUBJECT_TARGET_SHUFFLE_OPTIONS:
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

    if true_subject_slot is None:
        true_subject_slot = "subject0"
    if true_target_slot is None:
        true_target_slot = "target0"

    st_result = SUBJECT_TARGET_AGENT.run(
        text=inp.get("text", ""),
        audio_caption=inp.get("audio_caption", ""),
        subject_slots=subject_slots,
        target_slots=target_slots,
        scenario=given_scenario,
        domain=given_domain,
        culture=given_culture,
    )

    # 涓嶄紶鍏?scenario/domain/culture
    user_text = f"Text: {inp.get('text', '')}\n"
    if 'audio_caption' in inp and inp['audio_caption']:
        user_text += f"Audio Caption: {inp['audio_caption']}\n"

    user_text += "\n\n" + SUBJECT_TARGET_GROUNDING_STEPS.strip()
    user_text += "\n\n" + SUBJECT_TARGET_AGENT.build_prompt_block(st_result, include_recommendation=False)
    user_text += "\n\nSubject options (choose exact text for JSON 'subject'):\n" + "\n".join([f"- {v}" for v in subject_slots.values()])
    user_text += "\n\nTarget options (choose exact text for JSON 'target'):\n" + "\n".join([f"- {v}" for v in target_slots.values()])
    user_text += "\n\nCRITICAL REMINDER: For 'subject' and 'target', output exact entity text from options above. Do NOT output placeholder IDs."
    user_text += "\n\nProvide the 4-field JSON response."

    content_blocks = [{"type": "text", "text": user_text}]
    has_frame_folder = bool(inp.get("url_frame"))
    is_video = has_frame_folder or ('media_path_local' in inp) or str(inp.get('path', '')).endswith('.mp4')
    try:
        if is_video:
            if has_frame_folder:
                frame_count_raw = inp.get("frame_count", 0)
                try:
                    frame_count = int(frame_count_raw)
                except Exception:
                    frame_count = 0
                if frame_count <= 0:
                    frame_count = 4
                frame_count = min(frame_count, MAX_IMAGES_PER_PROMPT)

                frame_urls = build_frame_urls_from_folder(inp.get("url_frame"), str(sample_id), frame_count)
                if not frame_urls:
                    return {"id": sample_id, "error": "No frame urls generated from url_frame", "token_usage": token_usage}

                for fu in frame_urls:
                    content_blocks.append({"type": "image_url", "image_url": {"url": fu}})
            else:
                # Backward compatibility: fallback to local video frame extraction
                vid_path = inp.get('media_path_local') or inp.get('path')
                frames = extract_video_frames_4(vid_path, Path("./_temp_eval_frames"), sample_id, RANDOM_SEED)
                if len(frames) < 4:
                    print(f"[FrameWarning] id={sample_id} extracted_frames={len(frames)}/4")
                for f in frames:
                    content_blocks.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(f)}})
        else:
            img_path = inp.get('path')
            content_blocks.append({"type": "image_url", "image_url": {"url": local_image_to_data_url(img_path)}})
    except Exception as e:
        return {"id": sample_id, "error": f"Media error: {e}", "token_usage": token_usage}

    sit_norm = normalize_text(given_scenario)

    if sit_norm == "intent":
        global INTENT_SOTA_AGENT
        if INTENT_SOTA_AGENT is None:
            INTENT_SOTA_AGENT = build_intent_sota_agent(client)

        grounding_context = SUBJECT_TARGET_AGENT.build_prompt_block(st_result, include_recommendation=True)
        intent_pack = INTENT_SOTA_AGENT.run(
            sample_id=str(sample_id),
            text=inp.get("text", ""),
            audio_caption=inp.get("audio_caption", ""),
            subject_slots=subject_slots,
            target_slots=target_slots,
            media_blocks=content_blocks[1:] if len(content_blocks) > 1 else [],
            grounding_context=grounding_context,
        )
        intent_usage = intent_pack.get("token_usage", {})
        token_usage["prompt_tokens"] += int(intent_usage.get("prompt_tokens", 0))
        token_usage["completion_tokens"] += int(intent_usage.get("completion_tokens", 0))
        token_usage["total_tokens"] += int(intent_usage.get("total_tokens", 0))
        token_usage["cached_tokens"] += int(intent_usage.get("cached_tokens", 0))

        intent_pred = intent_pack.get("prediction", {}) if isinstance(intent_pack, dict) else {}
        final_pred_sub_slot = str(intent_pred.get("subject_slot", "subject0") or "subject0")
        final_pred_tgt_slot = str(intent_pred.get("target_slot", "target0") or "target0")
        if final_pred_sub_slot not in subject_slots:
            final_pred_sub_slot = "subject0"
        if final_pred_tgt_slot not in target_slots:
            final_pred_tgt_slot = "target0"

        final_pred_sub = subject_slots.get(final_pred_sub_slot, "")
        final_pred_tgt = target_slots.get(final_pred_tgt_slot, "")

        prediction = {
            "mechanism": str(intent_pred.get("mechanism", "")),
            "label": str(intent_pred.get("label", "")),
            "subject": final_pred_sub,
            "target": final_pred_tgt,
        }

        raw_pred_mech = prediction.get("mechanism", "")
        raw_pred_label = prediction.get("label", "")
        raw_pred_sub = final_pred_sub
        raw_pred_tgt = final_pred_tgt

        mech_norm = normalize_taxonomy_value(raw_pred_mech)
        label_norm = normalize_taxonomy_value(raw_pred_label)
        allowed_mechs = [normalize_taxonomy_value(m) for m in VALID_MECHANISMS.get(sit_norm, [])]
        allowed_labels = [normalize_taxonomy_value(l) for l in VALID_LABELS.get(sit_norm, [])]

        error_analysis = {
            "mech_mismatch": mech_norm not in allowed_mechs if allowed_mechs else True,
            "label_mismatch": label_norm not in allowed_labels if allowed_labels else True,
            "subject_format_error": False,
            "target_format_error": False,
            "predicted_mechanism": raw_pred_mech,
            "predicted_label": raw_pred_label,
            "predicted_subject": raw_pred_sub,
            "predicted_target": raw_pred_tgt,
            "final_subject": final_pred_sub,
            "final_target": final_pred_tgt,
            "final_subject_slot": final_pred_sub_slot,
            "final_target_slot": final_pred_tgt_slot,
            "subject_target_resolution": {"source": "intent_sota_pipeline"},
            "role_probe_error": None,
            "baseline_role_probe_error": None,
        }

        fields = ["mechanism", "label", "subject", "target"]
        matches = {}
        is_strict = True
        for f in fields:
            if f in ["subject", "target"]:
                gt_text = out.get(f, "")
                m = (normalize_slot(prediction.get(f, "")) == normalize_slot(gt_text))
            else:
                m = (normalize_taxonomy_value(prediction.get(f, "")) == normalize_taxonomy_value(out.get(f, "")))
            matches[f] = m
            if not m:
                is_strict = False

        return {
            "id": sample_id,
            "ground_truth": {f: out.get(f) for f in fields},
            "prediction": prediction,
            "matches": matches,
            "strict_match": is_strict,
            "error_analysis": error_analysis,
            "meta_scenario": given_scenario,
            "meta_domain": out.get("domain"),
            "meta_culture": out.get("culture"),
            "token_usage": token_usage,
            "slot_maps": {
                "subject_slots": subject_slots,
                "target_slots": target_slots,
                "true_subject_slot": true_subject_slot,
                "true_target_slot": true_target_slot,
            },
            "subject_target_agent": {
                "pipeline_result": st_result.as_dict(),
                "resolution": {"source": "intent_sota_pipeline"},
                "role_probe": {
                    "enabled": False,
                    "primary": None,
                    "extra_candidates": [],
                    "traces": [],
                    "error": None,
                },
            },
            "intent_sota": intent_pack.get("debug", {}),
        }
    role_probe_result = None
    role_probe_extra_candidates = []
    role_probe_trace = []
    role_probe_error = None
    if SUBJECT_TARGET_ROLE_PROBE_ENABLED:
        for probe_mode in ["literal", "pragmatic"]:
            try:
                probe_system_prompt, probe_user_prompt = SUBJECT_TARGET_AGENT.build_role_probe_prompts(
                    scenario=given_scenario,
                    text=inp.get("text", ""),
                    audio_caption=inp.get("audio_caption", ""),
                    subject_slots=subject_slots,
                    target_slots=target_slots,
                    result=st_result,
                    mode=probe_mode,
                )
                probe_content_blocks = [{"type": "text", "text": probe_user_prompt}]
                if len(content_blocks) > 1:
                    probe_content_blocks.extend(content_blocks[1:])

                probe_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": probe_system_prompt},
                        {"role": "user", "content": probe_content_blocks},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                probe_usage = extract_usage_from_response(probe_response)
                token_usage["prompt_tokens"] += probe_usage["prompt_tokens"]
                token_usage["completion_tokens"] += probe_usage["completion_tokens"]
                token_usage["total_tokens"] += probe_usage["total_tokens"]
                token_usage["cached_tokens"] += probe_usage["cached_tokens"]

                probe_content = probe_response.choices[0].message.content
                probe_raw = json.loads(probe_content) if probe_content else {}
                if not isinstance(probe_raw, dict):
                    probe_raw = {}
                probe_parsed = SUBJECT_TARGET_AGENT.parse_role_probe_prediction(
                    probe_raw,
                    subject_slots=subject_slots,
                    target_slots=target_slots,
                )
                role_probe_trace.append(
                    {
                        "mode": probe_mode,
                        "raw": probe_raw,
                        "parsed": probe_parsed.as_dict(),
                        "error": None,
                    }
                )
                if role_probe_result is None:
                    role_probe_result = probe_parsed
                else:
                    role_probe_extra_candidates.append(
                        {
                            "name": f"role_probe_{probe_mode}",
                            "subject": probe_parsed.subject_slot,
                            "target": probe_parsed.target_slot,
                            "confidence": probe_parsed.confidence,
                        }
                    )
            except Exception as e:
                err = str(e)
                role_probe_trace.append({"mode": probe_mode, "raw": {}, "parsed": None, "error": err})
                role_probe_error = err

    baseline_role_probe_error = None
    if SUBJECT_TARGET_BASELINE_ROLE_PROBE_ENABLED:
        try:
            baseline_system_prompt, baseline_user_prompt = build_baseline_role_probe_prompts(
                scenario=given_scenario,
                text=inp.get("text", ""),
                audio_caption=inp.get("audio_caption", ""),
                subject_slots=subject_slots,
                target_slots=target_slots,
            )
            baseline_content_blocks = [{"type": "text", "text": baseline_user_prompt}]
            if len(content_blocks) > 1:
                baseline_content_blocks.extend(content_blocks[1:])

            baseline_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": baseline_system_prompt},
                    {"role": "user", "content": baseline_content_blocks},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            baseline_usage = extract_usage_from_response(baseline_response)
            token_usage["prompt_tokens"] += baseline_usage["prompt_tokens"]
            token_usage["completion_tokens"] += baseline_usage["completion_tokens"]
            token_usage["total_tokens"] += baseline_usage["total_tokens"]
            token_usage["cached_tokens"] += baseline_usage["cached_tokens"]

            baseline_content = baseline_response.choices[0].message.content
            baseline_raw = json.loads(baseline_content) if baseline_content else {}
            if not isinstance(baseline_raw, dict):
                baseline_raw = {}
            baseline_parsed = SUBJECT_TARGET_AGENT.parse_role_probe_prediction(
                baseline_raw,
                subject_slots=subject_slots,
                target_slots=target_slots,
            )
            role_probe_trace.append(
                {
                    "mode": "baseline_direct",
                    "raw": baseline_raw,
                    "parsed": baseline_parsed.as_dict(),
                    "error": None,
                }
            )
            if (
                role_probe_result is None
                and baseline_parsed.subject_slot is not None
                and baseline_parsed.target_slot is not None
            ):
                role_probe_result = baseline_parsed
            role_probe_extra_candidates.append(
                {
                    "name": "baseline_role_probe",
                    "subject": baseline_parsed.subject_slot,
                    "target": baseline_parsed.target_slot,
                    "confidence": max(
                        float(baseline_parsed.confidence),
                        float(SUBJECT_TARGET_BASELINE_ROLE_PROBE_MIN_CONF),
                    ),
                }
            )
        except Exception as e:
            baseline_role_probe_error = str(e)
            role_probe_trace.append({"mode": "baseline_direct", "raw": {}, "parsed": None, "error": baseline_role_probe_error})

    prediction = None
    api_error = None
    raw_pred_mech = ""
    raw_pred_label = ""
    raw_pred_sub = ""
    raw_pred_tgt = ""
    for attempt in range(MAX_RETRIES):
        try:
            sit_norm = normalize_text(given_scenario)
            if sit_norm == "affection":
                system_prompt = SYSTEM_PROMPT_AFFECTION
            elif sit_norm == "attitude":
                system_prompt = SYSTEM_PROMPT_ATTITUDE
            elif sit_norm == "intent":
                system_prompt = SYSTEM_PROMPT_INTENT
            else:
                return {"id": sample_id, "error": f"Unknown scenario: {given_scenario}", "token_usage": token_usage}

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content_blocks}],
                response_format={"type": "json_object"}, temperature=0.0
            )
            one_usage = extract_usage_from_response(response)
            token_usage["prompt_tokens"] += one_usage["prompt_tokens"]
            token_usage["completion_tokens"] += one_usage["completion_tokens"]
            token_usage["total_tokens"] += one_usage["total_tokens"]
            token_usage["cached_tokens"] += one_usage["cached_tokens"]

            content = response.choices[0].message.content
            if not content:
                api_error = "Empty response content from model."
                time.sleep(1)
                continue

            prediction = json.loads(content)
            if not isinstance(prediction, dict):
                api_error = "Model output is not a JSON object."
                prediction = None
                time.sleep(1)
                continue

            # Record raw predicted fields for downstream mismatch analysis.
            raw_pred_mech = prediction.get("mechanism", "")
            raw_pred_label = prediction.get("label", "")
            raw_pred_sub = prediction.get("subject", "")
            raw_pred_tgt = prediction.get("target", "")

            api_error = None
            break
        except Exception as e:
            api_error = str(e)
            time.sleep(2)

    if api_error or prediction is None:
        return {"id": sample_id, "error": api_error or "No valid prediction generated.", "token_usage": token_usage}

    final_pred_sub_slot, final_pred_tgt_slot, st_resolution = SUBJECT_TARGET_AGENT.resolve(
        raw_subject=raw_pred_sub,
        raw_target=raw_pred_tgt,
        subject_slots=subject_slots,
        target_slots=target_slots,
        result=st_result,
        probe_result=role_probe_result,
        extra_candidates=role_probe_extra_candidates,
        text=inp.get("text", ""),
        audio_caption=inp.get("audio_caption", ""),
    )
    final_pred_sub = subject_slots.get(final_pred_sub_slot, str(raw_pred_sub or ""))
    final_pred_tgt = target_slots.get(final_pred_tgt_slot, str(raw_pred_tgt or ""))
    prediction["subject"] = final_pred_sub
    prediction["target"] = final_pred_tgt

    # 閿欒鍒嗘瀽锛氬熀浜庣粰瀹氱殑 scenario 鏍￠獙鏈哄埗鍜屾爣绛炬槸鍚﹀悎娉?(瀵归綈 Qwen 5. 閮ㄥ垎閫昏緫)
    sit_norm = normalize_text(given_scenario)
    mech_norm = normalize_taxonomy_value(raw_pred_mech)
    label_norm = normalize_taxonomy_value(raw_pred_label)
    allowed_mechs = [normalize_taxonomy_value(m) for m in VALID_MECHANISMS.get(sit_norm, [])]
    allowed_labels = [normalize_taxonomy_value(l) for l in VALID_LABELS.get(sit_norm, [])]

    valid_subject_norm = {normalize_slot(v) for v in subject_slots.values()}
    valid_target_norm = {normalize_slot(v) for v in target_slots.values()}
    raw_subject_norm = normalize_slot(raw_pred_sub)
    raw_target_norm = normalize_slot(raw_pred_tgt)
    error_analysis = {
        "mech_mismatch": mech_norm not in allowed_mechs if allowed_mechs else True,
        "label_mismatch": label_norm not in allowed_labels if allowed_labels else True,
        "subject_format_error": raw_subject_norm not in valid_subject_norm and raw_subject_norm not in {"subject0", "subject1", "subject2", "subject3"},
        "target_format_error": raw_target_norm not in valid_target_norm and raw_target_norm not in {"target0", "target1", "target2", "target3"},
        "predicted_mechanism": raw_pred_mech, "predicted_label": raw_pred_label,
        "predicted_subject": raw_pred_sub, "predicted_target": raw_pred_tgt,
        "final_subject": final_pred_sub, "final_target": final_pred_tgt,
        "final_subject_slot": final_pred_sub_slot,
        "final_target_slot": final_pred_tgt_slot,
        "subject_target_resolution": st_resolution,
        "role_probe_error": role_probe_error,
        "baseline_role_probe_error": baseline_role_probe_error,
    }

    # 鍒ゅ畾鍒ゅ畾
    fields = ["mechanism", "label", "subject", "target"]
    matches = {}
    is_strict = True
    for f in fields:
        if f in ["subject", "target"]:
            gt_text = out.get(f, "")
            m = (normalize_slot(prediction.get(f, "")) == normalize_slot(gt_text))
        else:
            m = (normalize_taxonomy_value(prediction.get(f, "")) == normalize_taxonomy_value(out.get(f, "")))
        matches[f] = m
        if not m: is_strict = False

    return {
        "id": sample_id,
        "ground_truth": {f: out.get(f) for f in fields},
        "prediction": prediction, "matches": matches, "strict_match": is_strict, "error_analysis": error_analysis,
        "meta_scenario": given_scenario, "meta_domain": out.get("domain"), "meta_culture": out.get("culture"),
        "token_usage": token_usage,
        "slot_maps": {"subject_slots": subject_slots, "target_slots": target_slots,
                      "true_subject_slot": true_subject_slot, "true_target_slot": true_target_slot},
        "subject_target_agent": {
            "pipeline_result": st_result.as_dict(),
            "resolution": st_resolution,
            "role_probe": {
                "enabled": SUBJECT_TARGET_ROLE_PROBE_ENABLED,
                "primary": role_probe_result.as_dict() if role_probe_result is not None else None,
                "extra_candidates": role_probe_extra_candidates,
                "traces": role_probe_trace,
                "error": role_probe_error,
            },
        },
    }


# ==========================================
# 4. 鎸囨爣璁＄畻涓庡彲瑙嗗寲 (瀵归綈 Qwen 鏍囧噯鍖栭娴嬪垎 + Strict All鏌卞瓙)
# ==========================================
def calculate_metrics_for_subset(results_list):
    if not results_list: return {"Count": 0}
    metrics = {"Count": len(results_list)}
    metrics["Strict_All4_Acc"] = accuracy_score_local(
        [1] * len(results_list),
        [1 if r['strict_match'] else 0 for r in results_list]
    )

    fields = ["mechanism", "label", "subject", "target"]
    for f in fields:
        y_true = [normalize_text(r['ground_truth'].get(f, "")) for r in results_list]
        y_pred = [normalize_text(r['prediction'].get(f, "")) for r in results_list]
        metrics[f"{f}_Accuracy"] = accuracy_score_local(y_true, y_pred)
        metrics[f"{f}_F1"] = macro_f1_score_local(y_true, y_pred)
    return metrics


def plot_metrics(report, output_path, successful_count, model_name):
    if plt is None:
        print("[PlotWarning] matplotlib is not available. Skip metrics plot.")
        return

    overall = report[f"1. Overall ({successful_count} Successful Samples)"]
    # 浠呯粯鍒舵ā鍨嬬湡瀹為娴嬬殑 4 涓瓧娈碉紝浠ュ強 Strict Match (瀵归綈 Qwen 閫昏緫)
    plot_fields = ["mechanism", "label", "subject", "target", "strict_all"]
    accs = [overall.get(f"{f}_Accuracy", overall.get('Strict_All4_Acc') if f == "strict_all" else 0) for f in
            plot_fields]
    f1s = [overall.get(f"{f}_F1", 0) for f in plot_fields]

    x = list(range(len(plot_fields)))
    x_left = [v - 0.35 / 2 for v in x]
    x_right = [v + 0.35 / 2 for v in x]
    width = 0.35
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_left, accs, width, label='Accuracy', color='#2b83ba', edgecolor='black', linewidth=0.7)
    ax.bar(x_right, f1s, width, label='Macro F1', color='#d7191c', edgecolor='black', linewidth=0.7)
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
# 5. 涓诲共閫昏緫 (澧炲姞绗?5 閮ㄥ垎閿欒鎶ュ憡)
# ==========================================
def run_evaluation():
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    combined = adapt_samples_to_legacy(loaded)

    if not combined:
        print("No valid samples loaded.")
        return

    rng = random.Random(RANDOM_SEED)
    grouped = {}
    for item in combined:
        scenario = normalize_text(item.get("output", {}).get("scenario", item.get("output", {}).get("situation", "")))
        grouped.setdefault(scenario, []).append(item)
    sampled = []
    for scenario in [normalize_text(x) for x in SCENARIOS]:
        items = grouped.get(scenario, [])
        if not items:
            continue
        if SAMPLE_SIZE and SAMPLE_SIZE > 0:
            sampled.extend(rng.sample(items, min(SAMPLE_SIZE, len(items))))
        else:
            sampled.extend(items)
    if not SILICONFLOW_API_KEY:
        raise ValueError("Missing SILICONFLOW_API_KEY (or OPENAI_API_KEY) in environment.")
    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

    global INTENT_SOTA_AGENT
    INTENT_SOTA_AGENT = build_intent_sota_agent(client)

    results, failed = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, s, client): s for s in sampled}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                res = f.result()
            except Exception as e:
                sample = futures.get(f, {})
                sample_inp = sample.get("input", {}) if isinstance(sample, dict) else {}
                sid = sample_inp.get("id") or sample_inp.get("samples_id")
                failed.append({
                    "id": sid,
                    "error": f"Worker exception: {e}",
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
                })
                print(f"[Failed] Worker exception: {e}")
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(sampled)}")
                continue
            if "error" not in res:
                results.append(res)
            else:
                failed.append(res); print(f"[Failed] {res['error']}")
            if i % 10 == 0: print(f"Progress: {i}/{len(sampled)}")

    with open(OUTPUT_DETAILED_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_FAILED_FILE, 'w', encoding='utf-8') as f:
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
    with open(OUTPUT_TOKEN_USAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(token_usage_rows, f, indent=2, ensure_ascii=False)

    if not results: return
    report = {"1. Overall": calculate_metrics_for_subset(results)}
    for group_name, key in [("2. By Scenario", "meta_scenario"), ("3. By Domain", "meta_domain"),
                            ("4. By Culture", "meta_culture")]:
        report[group_name] = {val: calculate_metrics_for_subset([r for r in results if r[key] == val]) for val in
                              globals()[group_name.split()[-1].upper() + 'S']}

    # 绗?5 閮ㄥ垎锛氶敊璇垎鏋愭姤鍛?(瀵归綈 Qwen)
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

    report[f"1. Overall ({len(results)} Successful Samples)"] = report.pop("1. Overall")
    report["5. Error Analysis"] = err_rep

    with open(OUTPUT_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    plot_metrics(report, OUTPUT_PLOT_FILE, len(results), MODEL_NAME)
    print(f"Done! Strict Acc: {report[f'1. Overall ({len(results)} Successful Samples)']['Strict_All4_Acc']:.2%}")
    print(f"Saved token usage file: {OUTPUT_TOKEN_USAGE_FILE}")


FORMULA_FIELDS = ["label", "subject", "target", "mechanism"]


def _formula_safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _formula_mean_optional(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _formula_get_scenario(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("meta_scenario", record.get("meta_situation", "")))


def _formula_compute_match_fallback(record: Dict[str, Any], field: str) -> bool:
    gt = normalize_text(record.get("ground_truth", {}).get(field, ""))
    pred = normalize_text(record.get("prediction", {}).get(field, ""))
    return pred == gt


def _formula_get_match(record: Dict[str, Any], field: str) -> bool:
    matches = record.get("matches", {})
    if isinstance(matches, dict) and field in matches:
        return bool(matches[field])
    return _formula_compute_match_fallback(record, field)


def _formula_compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "N": 0,
            "independent_accuracy": {f: None for f in FORMULA_FIELDS},
            "conditional_accuracy": {
                "mechanism_given_label": None,
                "denominator_label_correct": 0,
            },
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
        "n_label_correct": metrics["conditional_accuracy"].get("denominator_label_correct", 0),
        "indep_components": metrics.get("independent_accuracy", {}),
    }


def _formula_build_table_ready(report: Dict[str, Any], scenario_order: List[str], method_name: str) -> Dict[str, Any]:
    order = scenario_order + ["overall"]
    raw_values: Dict[str, Dict[str, Any]] = {}
    pct_values: Dict[str, Dict[str, Any]] = {}
    for key in order:
        metrics = report["overall"] if key == "overall" else report["by_scenario"][key]
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


def _iter_detail_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        for item in payload["data"]:
            if isinstance(item, dict):
                yield item


def compute_formula_metrics_from_detail(detail_path: Path) -> None:
    if not detail_path.exists():
        print(f"[Formula] Skip: detail file not found: {detail_path}")
        return

    raw = json.loads(detail_path.read_text(encoding="utf-8"))
    records = list(_iter_detail_records(raw))
    scenario_order = [normalize_text(x) for x in SCENARIOS if normalize_text(x)]
    if not scenario_order:
        scenario_order = ["affection", "attitude", "intent"]

    by_scenario: Dict[str, List[Dict[str, Any]]] = {s: [] for s in scenario_order}
    for rec in records:
        s = _formula_get_scenario(rec)
        if s in by_scenario:
            by_scenario[s].append(rec)

    report = {
        "input_file": str(detail_path),
        "total_records": len(records),
        "formula_note": {
            "independent_accuracy": "Acc_y for y in {Label, Subject, Target, Mechanism}",
            "table_indep": "Indep = mean(Acc_Label, Acc_Subject, Acc_Target, Acc_Mechanism)",
            "conditional_accuracy": "Acc_M|L = P(Mechanism correct | Label correct)",
            "relational_alignment": "Acc_S^T = P(Subject and Target both correct)",
            "joint_accuracy": "Acc_Joint = P(Label, Subject, Target, Mechanism all correct)",
        },
        "overall": _formula_compute_metrics(records),
        "by_scenario": {s: _formula_compute_metrics(by_scenario[s]) for s in scenario_order},
    }
    report["table_ready"] = _formula_build_table_ready(report, scenario_order, MODEL_NAME)

    with open(OUTPUT_FORMULA_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_FORMULA_TABLE_FILE, "w", encoding="utf-8") as f:
        json.dump(report["table_ready"], f, indent=2, ensure_ascii=False)

    print(f"[Formula] Saved metrics report: {OUTPUT_FORMULA_METRICS_FILE}")
    print(f"[Formula] Saved table-ready report: {OUTPUT_FORMULA_TABLE_FILE}")


def refresh_output_paths() -> None:
    global OUTPUT_DIR
    global OUTPUT_DETAILED_FILE
    global OUTPUT_METRICS_FILE
    global OUTPUT_FAILED_FILE
    global OUTPUT_TOKEN_USAGE_FILE
    global OUTPUT_PLOT_FILE
    global OUTPUT_FORMULA_METRICS_FILE
    global OUTPUT_FORMULA_TABLE_FILE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DETAILED_FILE = OUTPUT_DIR / "evaluation_predictions_detailed.json"
    OUTPUT_METRICS_FILE = OUTPUT_DIR / "evaluation_metrics_report.json"
    OUTPUT_FAILED_FILE = OUTPUT_DIR / "evaluation_failures.json"
    OUTPUT_TOKEN_USAGE_FILE = OUTPUT_DIR / "evaluation_token_usage_by_id.json"
    OUTPUT_PLOT_FILE = OUTPUT_DIR / "evaluation_metrics_plot.png"
    OUTPUT_FORMULA_METRICS_FILE = OUTPUT_DIR / "evaluation_formula_metrics.json"
    OUTPUT_FORMULA_TABLE_FILE = OUTPUT_DIR / "evaluation_formula_metrics_table_ready.json"


def apply_cli_args() -> None:
    parser = argparse.ArgumentParser(description="CoDAR v2 evaluator with intent SOTA multi-stage pipeline")
    parser.add_argument("--input-json-path", dest="input_json_path", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--max-workers", dest="max_workers", type=int, default=None)
    parser.add_argument("--sample-size", dest="sample_size", type=int, default=None)

    parser.add_argument("--stage-a-votes", dest="stage_a_votes", type=int, default=None)
    parser.add_argument("--stage-a-top-k", dest="stage_a_top_k", type=int, default=None)
    parser.add_argument("--stage-a-alt-weight", dest="stage_a_alt_weight", type=float, default=None)
    parser.add_argument("--stage-c-votes", dest="stage_c_votes", type=int, default=None)
    parser.add_argument("--label-signal-conf-threshold", dest="label_signal_conf_threshold", type=float, default=None)
    parser.add_argument("--hard-refine-top-conf-threshold", dest="hard_refine_top_conf_threshold", type=float, default=None)
    parser.add_argument("--disable-prior-router", dest="disable_prior_router", action="store_true")
    parser.add_argument("--prior-base-file", dest="prior_base_file", default=None)
    parser.add_argument("--prior-cot-file", dest="prior_cot_file", default=None)
    parser.add_argument("--prior-mech-threshold", dest="prior_mech_threshold", type=float, default=None)
    parser.add_argument("--prior-base-only-mech-threshold", dest="prior_base_only_mech_threshold", type=float, default=None)
    parser.add_argument("--prior-label-threshold", dest="prior_label_threshold", type=float, default=None)

    args = parser.parse_args()

    global INPUT_JSON_PATH
    global OUTPUT_DIR
    global MODEL_NAME
    global MAX_WORKERS
    global SAMPLE_SIZE
    global INTENT_SOTA_STAGE_A_VOTES
    global INTENT_SOTA_STAGE_A_TOP_K
    global INTENT_SOTA_STAGE_A_ALT_WEIGHT
    global INTENT_SOTA_STAGE_C_VOTES
    global INTENT_SOTA_LABEL_SIGNAL_CONF_THRESHOLD
    global INTENT_SOTA_HARD_REFINE_TOP_CONF_THRESHOLD
    global INTENT_SOTA_DISABLE_PRIOR_ROUTER
    global INTENT_SOTA_PRIOR_BASE_FILE
    global INTENT_SOTA_PRIOR_COT_FILE
    global INTENT_SOTA_PRIOR_MECH_THRESHOLD
    global INTENT_SOTA_PRIOR_BASE_ONLY_MECH_THRESHOLD
    global INTENT_SOTA_PRIOR_LABEL_THRESHOLD

    if args.input_json_path:
        INPUT_JSON_PATH = str(args.input_json_path)
    if args.output_dir:
        OUTPUT_DIR = Path(str(args.output_dir))
        refresh_output_paths()
    if args.model_name:
        MODEL_NAME = str(args.model_name)
    if args.max_workers is not None:
        MAX_WORKERS = max(1, int(args.max_workers))
    if args.sample_size is not None:
        SAMPLE_SIZE = max(0, int(args.sample_size))

    if args.stage_a_votes is not None:
        INTENT_SOTA_STAGE_A_VOTES = max(1, int(args.stage_a_votes))
    if args.stage_a_top_k is not None:
        INTENT_SOTA_STAGE_A_TOP_K = max(1, int(args.stage_a_top_k))
    if args.stage_a_alt_weight is not None:
        INTENT_SOTA_STAGE_A_ALT_WEIGHT = float(args.stage_a_alt_weight)
    if args.stage_c_votes is not None:
        INTENT_SOTA_STAGE_C_VOTES = max(1, int(args.stage_c_votes))
    if args.label_signal_conf_threshold is not None:
        INTENT_SOTA_LABEL_SIGNAL_CONF_THRESHOLD = float(args.label_signal_conf_threshold)
    if args.hard_refine_top_conf_threshold is not None:
        INTENT_SOTA_HARD_REFINE_TOP_CONF_THRESHOLD = float(args.hard_refine_top_conf_threshold)

    if args.disable_prior_router:
        INTENT_SOTA_DISABLE_PRIOR_ROUTER = True
    if args.prior_base_file is not None:
        INTENT_SOTA_PRIOR_BASE_FILE = str(args.prior_base_file)
    if args.prior_cot_file is not None:
        INTENT_SOTA_PRIOR_COT_FILE = str(args.prior_cot_file)
    if args.prior_mech_threshold is not None:
        INTENT_SOTA_PRIOR_MECH_THRESHOLD = float(args.prior_mech_threshold)
    if args.prior_base_only_mech_threshold is not None:
        INTENT_SOTA_PRIOR_BASE_ONLY_MECH_THRESHOLD = float(args.prior_base_only_mech_threshold)
    if args.prior_label_threshold is not None:
        INTENT_SOTA_PRIOR_LABEL_THRESHOLD = float(args.prior_label_threshold)


def main():
    apply_cli_args()
    run_evaluation()
    compute_formula_metrics_from_detail(Path(OUTPUT_DETAILED_FILE))


if __name__ == "__main__":
    main()

