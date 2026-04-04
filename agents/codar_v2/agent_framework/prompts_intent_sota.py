"""Prompt templates and taxonomy constants for intent-focused SOTA pipeline."""

from __future__ import annotations

INTENT_MECHANISMS = [
    "expressive_aggression",
    "malicious_manipulation",
    "prosocial_deception",
    "benevolent_provocation",
]

INTENT_LABELS = [
    "mitigate",
    "intimidate",
    "alienate",
    "mock",
    "denounce",
    "provoke",
    "dominate",
    "condemn",
]

SLOT_KEYS = ["subject0", "subject1", "subject2", "subject3"]


MECHANISM_BOUNDARY_CARD = """
Mechanism boundary rules (strict):
1) expressive_aggression: direct hostile/ridiculing/aggressive expression without a clear strategic cover story.
2) malicious_manipulation: strategic control through deception/coercion/threat to gain leverage.
3) prosocial_deception: socially soft/face-saving framing (white lie, polite cover, conflict-mitigation framing) to preserve harmony.
4) benevolent_provocation: deliberate challenge/teasing intended to motivate growth or corrective action.
5) If strategic intent is unclear, default to expressive_aggression (do NOT overuse minority mechanisms).
""".strip()


LABEL_BOUNDARY_CARD = """
Label boundary rules (strict):
- mitigate: reduce tension, soften criticism, restore cooperation.
- mock: make target look ridiculous for amusement.
- provoke: bait emotional reaction/escalation.
- alienate: cast target as outsider/stereotype, push social distance.
- denounce: public condemnation aimed at social accountability/punishment.
- condemn: moral judgment of wrongdoing (not necessarily public mobilization).
- dominate: assert hierarchy/control superiority.
- intimidate: induce fear/compliance via threat pressure.

Critical pair boundaries:
- mock vs provoke: amusement-centered -> mock; reaction-seeking escalation -> provoke.
- mock vs alienate: ridicule-as-joke -> mock; identity-based exclusion -> alienate.
- denounce vs condemn: public call-out/accountability -> denounce; moral disapproval statement -> condemn.
- dominate vs intimidate: status/control assertion -> dominate; fear/compliance pressure -> intimidate.
- mitigate vs prosocial-looking sarcasm: if de-escalation is not the real end-goal, do NOT pick mitigate.
""".strip()


STAGE_A_SYSTEM_PROMPT = f"""
You are Stage-A for intent decoding.
Goal: choose ONE mechanism and ONE label from the allowed lists, using contrastive reasoning.

Allowed mechanisms: {INTENT_MECHANISMS}
Allowed labels: {INTENT_LABELS}

{MECHANISM_BOUNDARY_CARD}

{LABEL_BOUNDARY_CARD}

Output strict JSON only, with keys:
- mechanism
- label
- alt_mechanism
- alt_label
- ridicule_signal
- reaction_bait_signal
- exclusion_signal
- deescalation_signal
- moral_callout_signal
- coercive_control_signal
- hierarchy_signal
- strategic_cover_signal
- growth_challenge_signal
- mechanism_boundary_note
- label_boundary_note
- confidence

Rules:
- mechanism and alt_mechanism must be in allowed mechanisms.
- label and alt_label must be in allowed labels.
- all *_signal fields must be true/false.
- confidence must be a number in [0, 1].
- Keep notes short (<= 35 words each).
- If evidence supports both mock and provoke, output primary + closest alternative explicitly.
- Avoid defaulting to expressive_aggression when strategic intent is explicitly indicated.
""".strip()


STAGE_B_CHECKER_SYSTEM_PROMPT = f"""
You are Stage-B checker for intent decoding.
You receive multiple candidate (mechanism, label) pairs from Stage-A.
Pick the single best pair that most strictly satisfies boundary rules.

Allowed mechanisms: {INTENT_MECHANISMS}
Allowed labels: {INTENT_LABELS}

{MECHANISM_BOUNDARY_CARD}

{LABEL_BOUNDARY_CARD}

Output strict JSON only, with keys:
- winner_mechanism
- winner_label
- runner_up_mechanism
- runner_up_label
- accepted
- rejection_reason

Rules:
- accepted is true/false.
- If accepted is false, winner_* still must be valid allowed values.
- Keep rejection_reason concise.
""".strip()


STAGE_C_SLOT_SYSTEM_PROMPT = """
You are Stage-C slot resolver for intent.
Pick exactly one subject slot and one target slot from the provided candidate mappings.

Mandatory decision protocol:
1) Subject selection:
- Choose the actor who is performing the intent expression/action.
- Prefer explicit speaker/author/actor evidence over inferred background entities.
2) Target selection:
- Choose the direct recipient/object of control, ridicule, provocation, condemnation, or mitigation.
- Prefer explicitly addressed/evaluated entities over bystanders or broad context groups.
3) Contrastive check:
- Identify one nearest alternative for subject and one for target, then reject briefly.

Output strict JSON only with keys:
- subject_slot
- target_slot
- alt_subject_slot
- alt_target_slot
- subject_reject_note
- target_reject_note
- confidence

Rules:
- subject_slot and alt_subject_slot must be one of subject0..subject3
- target_slot and alt_target_slot must be one of target0..target3
- confidence in [0, 1]
- Keep reject notes short (<= 25 words each).
""".strip()


STAGE_D_REPAIR_SYSTEM_PROMPT = f"""
You are Stage-D final repair arbiter.
You will receive possibly invalid final fields.
Repair to a valid final decision.

Allowed mechanisms: {INTENT_MECHANISMS}
Allowed labels: {INTENT_LABELS}
Allowed subject slots: subject0..subject3
Allowed target slots: target0..target3

Output strict JSON only with keys:
- mechanism
- label
- subject_slot
- target_slot
- repair_note

Rules:
- All fields must be valid allowed values.
- Keep repair_note short.
""".strip()


STAGE_E_HARD_REFINE_SYSTEM_PROMPT = f"""
You are Stage-E hard-case refiner for intent.
Your job: refine ONLY mechanism and label for uncertain samples.

Allowed mechanisms: {INTENT_MECHANISMS}
Allowed labels: {INTENT_LABELS}

{MECHANISM_BOUNDARY_CARD}

{LABEL_BOUNDARY_CARD}

Output strict JSON only with keys:
- mechanism
- label
- refine_note

Rules:
- mechanism must be in allowed mechanisms.
- label must be in allowed labels.
- Use only provided evidence and candidate sets; do not invent new classes.
- Keep refine_note short (<= 35 words).
""".strip()


def build_common_user_payload(
    *,
    sample_id: str,
    text: str,
    audio_caption: str,
    subject_slots: dict[str, str],
    target_slots: dict[str, str],
) -> str:
    lines = [
        f"sample_id: {sample_id}",
        f"text: {text or ''}",
    ]
    if audio_caption:
        lines.append(f"audio_caption: {audio_caption}")

    lines.append("subject slot mapping:")
    for k in SLOT_KEYS:
        if k in subject_slots:
            lines.append(f"- {k}: {subject_slots[k]}")

    lines.append("target slot mapping:")
    for k in SLOT_KEYS:
        tk = k.replace("subject", "target")
        if tk in target_slots:
            lines.append(f"- {tk}: {target_slots[tk]}")

    return "\n".join(lines)


def build_stage_a_user_prompt(common_payload: str) -> str:
    return (
        common_payload
        + "\n\nTask: decide mechanism and label for intent. "
        + "You must include one closest alternative for each and a short boundary reason."
    )


def build_stage_b_user_prompt(common_payload: str, candidates: list[dict]) -> str:
    return (
        common_payload
        + "\n\nStage-A candidates (sorted):\n"
        + "\n".join([f"- {c}" for c in candidates])
        + "\n\nTask: choose the best candidate pair under strict boundary rules."
    )


def build_stage_c_user_prompt(common_payload: str) -> str:
    return (
        common_payload
        + "\n\nTask: resolve subject_slot and target_slot with contrastive boundary reasoning. "
        + "Output slots only (do not output candidate text itself)."
    )


def build_stage_d_user_prompt(common_payload: str, draft: dict) -> str:
    return (
        common_payload
        + "\n\nDraft final decision (may be invalid):\n"
        + f"{draft}\n"
        + "Task: repair to valid final mechanism/label/subject_slot/target_slot."
    )


def build_stage_e_user_prompt(
    common_payload: str,
    *,
    current_draft: dict,
    candidate_mechanisms: list[str],
    candidate_labels: list[str],
    stage_a_signal: dict,
    stage_b_summary: dict,
) -> str:
    return (
        common_payload
        + "\n\nCurrent draft:\n"
        + f"{current_draft}\n"
        + "Candidate mechanisms:\n"
        + f"{candidate_mechanisms}\n"
        + "Candidate labels:\n"
        + f"{candidate_labels}\n"
        + "Stage-A signal summary:\n"
        + f"{stage_a_signal}\n"
        + "Stage-B checker summary:\n"
        + f"{stage_b_summary}\n"
        + "Task: choose final mechanism and label from candidate sets only."
    )
