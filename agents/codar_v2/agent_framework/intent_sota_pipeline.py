from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import prompts_intent_sota as p


MECH_CANON = [
    "prosocial_deception",
    "malicious_manipulation",
    "expressive_aggression",
    "benevolent_provocation",
]

LABEL_CANON = [
    "mitigate",
    "intimidate",
    "alienate",
    "mock",
    "denounce",
    "provoke",
    "dominate",
    "condemn",
]

MECH_DISPLAY = {
    "prosocial_deception": "prosocial deception",
    "malicious_manipulation": "malicious manipulation",
    "expressive_aggression": "expressive aggression",
    "benevolent_provocation": "benevolent provocation",
}

_MECH_ALIASES = {
    "prosocial deception": "prosocial_deception",
    "prosocial_deception": "prosocial_deception",
    "malicious manipulation": "malicious_manipulation",
    "malicious_manipulation": "malicious_manipulation",
    "expressive aggression": "expressive_aggression",
    "expressive_aggression": "expressive_aggression",
    "benevolent provocation": "benevolent_provocation",
    "benevolent_provocation": "benevolent_provocation",
}


@dataclass
class IntentSotaConfig:
    stage_a_votes: int = 5
    stage_a_top_k: int = 4
    stage_a_alt_weight: float = 0.35
    stage_c_votes: int = 5
    label_signal_conf_threshold: float = 0.58
    hard_refine_top_conf_threshold: float = 0.72
    disable_prior_router: bool = False
    prior_base_file: str = ""
    prior_cot_file: str = ""
    prior_mech_threshold: float = 0.66
    prior_base_only_mech_threshold: float = 0.76
    prior_label_threshold: float = 0.68


def _clip01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").split())


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    sub = text[start : end + 1]
    try:
        obj = json.loads(sub)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _usage_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion) or (prompt + completion))
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
    }


def _sum_usage(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k in ["prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"]:
        dst[k] = int(dst.get(k, 0)) + int(src.get(k, 0))


def _canon_mech(value: Any) -> str:
    key = _norm(value)
    if key in _MECH_ALIASES:
        return _MECH_ALIASES[key]
    return ""


def _canon_label(value: Any) -> str:
    key = _norm(value)
    return key if key in LABEL_CANON else ""


def _slot_key(raw: Any, prefix: str) -> str:
    s = str(raw or "").strip().lower().replace(" ", "")
    if s in {f"{prefix}{i}" for i in range(4)}:
        return s
    return ""


def _tokenize(text: Any) -> List[str]:
    buf = []
    cur = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in "_'":
            cur.append(ch.lower())
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf


def _slot_evidence(merged_text: str, option: str) -> float:
    txt = _norm(merged_text)
    opt = _norm(option)
    if not opt:
        return 0.0
    if opt in txt:
        return 1.0
    tt = set(_tokenize(txt))
    ot = _tokenize(opt)
    if not ot:
        return 0.0
    hit = sum(1 for t in ot if t in tt)
    return hit / len(ot)


class IntentSotaPipeline:
    def __init__(self, *, client: Any, model_name: str, max_retries: int, config: IntentSotaConfig):
        self.client = client
        self.model_name = str(model_name)
        self.max_retries = int(max(1, max_retries))
        self.cfg = config
        self._prior_base = self._load_prior_file(config.prior_base_file)
        self._prior_cot = self._load_prior_file(config.prior_cot_file)

    def _load_prior_file(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        path = Path(str(file_path or "").strip())
        if not path:
            return {}
        if not path.exists() or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        out: Dict[str, Dict[str, Any]] = {}

        def add_row(row: Dict[str, Any]) -> None:
            if not isinstance(row, dict):
                return
            sid = str(row.get("id") or row.get("sample_id") or row.get("sample") or "").strip()
            if not sid:
                return

            mech = _canon_mech(
                row.get("mechanism")
                or row.get("pred_mechanism")
                or row.get("winner_mechanism")
                or row.get("top_mechanism")
            )
            label = _canon_label(
                row.get("label")
                or row.get("pred_label")
                or row.get("winner_label")
                or row.get("top_label")
            )

            mech_conf = _clip01(
                row.get("mechanism_confidence")
                or row.get("mech_conf")
                or row.get("mechanism_prob")
                or row.get("confidence")
                or 0.0
            )
            label_conf = _clip01(
                row.get("label_confidence")
                or row.get("lbl_conf")
                or row.get("label_prob")
                or row.get("confidence")
                or 0.0
            )

            if not mech and not label:
                return

            out[sid] = {
                "mechanism": mech,
                "label": label,
                "mechanism_confidence": mech_conf,
                "label_confidence": label_conf,
            }

        if isinstance(payload, list):
            for row in payload:
                add_row(row)
            return out

        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                for row in payload["data"]:
                    add_row(row)
                return out
            if isinstance(payload.get("records"), list):
                for row in payload["records"]:
                    add_row(row)
                return out
            for k, v in payload.items():
                if isinstance(v, dict):
                    merged = dict(v)
                    merged.setdefault("id", k)
                    add_row(merged)
            return out

        return out

    def _prior_lookup(self, sample_id: str) -> Dict[str, Any]:
        sid = str(sample_id or "").strip()
        return {
            "base": self._prior_base.get(sid, {}),
            "cot": self._prior_cot.get(sid, {}),
        }

    def _chat_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        media_blocks: List[Dict[str, Any]],
        temperature: float,
    ) -> Tuple[Dict[str, Any], Dict[str, int], str]:
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
        last_error = ""

        user_content = [{"type": "text", "text": user_text}]
        if media_blocks:
            user_content.extend(media_blocks)

        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=float(temperature),
                )
                one_usage = _usage_dict(getattr(response, "usage", None))
                _sum_usage(usage_total, one_usage)

                raw = ""
                if getattr(response, "choices", None):
                    choice = response.choices[0]
                    if getattr(choice, "message", None) is not None:
                        raw = str(choice.message.content or "")

                parsed = _extract_json_object(raw)
                if parsed:
                    return parsed, usage_total, ""
                last_error = "empty_or_invalid_json"
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.8)

        return {}, usage_total, last_error

    def _stage_a(
        self,
        *,
        common_payload: str,
        media_blocks: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        votes: List[Dict[str, Any]] = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

        vote_count = max(1, int(self.cfg.stage_a_votes))
        temps = [0.0] if vote_count == 1 else [min(0.8, 0.15 * i) for i in range(vote_count)]

        for i, temp in enumerate(temps):
            user_prompt = p.build_stage_a_user_prompt(common_payload)
            parsed, usage, err = self._chat_json(
                system_prompt=p.STAGE_A_SYSTEM_PROMPT,
                user_text=user_prompt,
                media_blocks=media_blocks,
                temperature=temp,
            )
            _sum_usage(usage_total, usage)

            mech = _canon_mech(parsed.get("mechanism"))
            label = _canon_label(parsed.get("label"))
            alt_mech = _canon_mech(parsed.get("alt_mechanism"))
            alt_label = _canon_label(parsed.get("alt_label"))
            conf = _clip01(parsed.get("confidence", 0.0))

            votes.append(
                {
                    "idx": i,
                    "temperature": temp,
                    "mechanism": mech,
                    "label": label,
                    "alt_mechanism": alt_mech,
                    "alt_label": alt_label,
                    "confidence": conf,
                    "signals": {
                        "ridicule_signal": bool(parsed.get("ridicule_signal", False)),
                        "reaction_bait_signal": bool(parsed.get("reaction_bait_signal", False)),
                        "exclusion_signal": bool(parsed.get("exclusion_signal", False)),
                        "deescalation_signal": bool(parsed.get("deescalation_signal", False)),
                        "moral_callout_signal": bool(parsed.get("moral_callout_signal", False)),
                        "coercive_control_signal": bool(parsed.get("coercive_control_signal", False)),
                        "hierarchy_signal": bool(parsed.get("hierarchy_signal", False)),
                        "strategic_cover_signal": bool(parsed.get("strategic_cover_signal", False)),
                        "growth_challenge_signal": bool(parsed.get("growth_challenge_signal", False)),
                    },
                    "mechanism_boundary_note": str(parsed.get("mechanism_boundary_note", "") or "").strip(),
                    "label_boundary_note": str(parsed.get("label_boundary_note", "") or "").strip(),
                    "error": err,
                    "raw": parsed,
                }
            )

        pair_score = defaultdict(float)
        pair_votes = Counter()
        mechanism_votes = Counter()
        label_votes = Counter()
        signal_counter = Counter()

        for row in votes:
            mech = row["mechanism"]
            label = row["label"]
            if mech and label:
                key = (mech, label)
                w = 1.0 + 0.4 * row["confidence"]
                pair_score[key] += w
                pair_votes[key] += 1
                mechanism_votes[mech] += 1
                label_votes[label] += 1

            am = row["alt_mechanism"]
            al = row["alt_label"]
            if am and al:
                akey = (am, al)
                pair_score[akey] += max(0.0, float(self.cfg.stage_a_alt_weight))

            for sig, val in row["signals"].items():
                if val:
                    signal_counter[sig] += 1

        ranked_pairs = sorted(pair_score.items(), key=lambda kv: (-kv[1], -pair_votes[kv[0]], kv[0][0], kv[0][1]))
        top_k = max(1, int(self.cfg.stage_a_top_k))
        top_candidates = [
            {
                "mechanism": k[0],
                "label": k[1],
                "score": round(v, 6),
                "votes": int(pair_votes[k]),
                "mechanism_votes": int(mechanism_votes.get(k[0], 0)),
                "label_votes": int(label_votes.get(k[1], 0)),
            }
            for k, v in ranked_pairs[:top_k]
        ]

        top_conf = 0.0
        if ranked_pairs:
            top_score = ranked_pairs[0][1]
            sum_score = sum(v for _, v in ranked_pairs)
            if sum_score > 0:
                top_conf = _clip01(top_score / sum_score)

        signals_majority = {}
        vlen = max(1, len(votes))
        for sig in [
            "ridicule_signal",
            "reaction_bait_signal",
            "exclusion_signal",
            "deescalation_signal",
            "moral_callout_signal",
            "coercive_control_signal",
            "hierarchy_signal",
            "strategic_cover_signal",
            "growth_challenge_signal",
        ]:
            signals_majority[sig] = signal_counter[sig] >= ((vlen + 1) // 2)

        stage = {
            "votes": votes,
            "top_candidates": top_candidates,
            "top_confidence": top_conf,
            "majority_signals": signals_majority,
            "mechanism_vote_counts": dict(mechanism_votes),
            "label_vote_counts": dict(label_votes),
        }
        return stage, usage_total

    def _stage_b(
        self,
        *,
        common_payload: str,
        media_blocks: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        if not candidates:
            fallback = {
                "winner_mechanism": "expressive_aggression",
                "winner_label": "provoke",
                "runner_up_mechanism": "expressive_aggression",
                "runner_up_label": "mock",
                "accepted": False,
                "rejection_reason": "no_stage_a_candidate",
            }
            return fallback, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

        user_prompt = p.build_stage_b_user_prompt(common_payload, candidates)
        parsed, usage, err = self._chat_json(
            system_prompt=p.STAGE_B_CHECKER_SYSTEM_PROMPT,
            user_text=user_prompt,
            media_blocks=media_blocks,
            temperature=0.0,
        )

        winner_mech = _canon_mech(parsed.get("winner_mechanism"))
        winner_label = _canon_label(parsed.get("winner_label"))
        runner_mech = _canon_mech(parsed.get("runner_up_mechanism"))
        runner_label = _canon_label(parsed.get("runner_up_label"))

        if not winner_mech or not winner_label:
            winner_mech = candidates[0]["mechanism"]
            winner_label = candidates[0]["label"]

        if not runner_mech or not runner_label:
            if len(candidates) > 1:
                runner_mech = candidates[1]["mechanism"]
                runner_label = candidates[1]["label"]
            else:
                runner_mech = winner_mech
                runner_label = winner_label

        return {
            "winner_mechanism": winner_mech,
            "winner_label": winner_label,
            "runner_up_mechanism": runner_mech,
            "runner_up_label": runner_label,
            "accepted": bool(parsed.get("accepted", True)),
            "rejection_reason": str(parsed.get("rejection_reason", err) or "").strip(),
            "raw": parsed,
            "error": err,
        }, usage

    def _stage_c(
        self,
        *,
        common_payload: str,
        media_blocks: List[Dict[str, Any]],
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        votes: List[Dict[str, Any]] = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
        vote_count = max(1, int(self.cfg.stage_c_votes))
        temps = [0.0] if vote_count == 1 else [min(0.6, 0.12 * i) for i in range(vote_count)]

        for i, temp in enumerate(temps):
            parsed, usage, err = self._chat_json(
                system_prompt=p.STAGE_C_SLOT_SYSTEM_PROMPT,
                user_text=p.build_stage_c_user_prompt(common_payload),
                media_blocks=media_blocks,
                temperature=temp,
            )
            _sum_usage(usage_total, usage)

            ss = _slot_key(parsed.get("subject_slot"), "subject")
            ts = _slot_key(parsed.get("target_slot"), "target")
            ass = _slot_key(parsed.get("alt_subject_slot"), "subject")
            ats = _slot_key(parsed.get("alt_target_slot"), "target")
            conf = _clip01(parsed.get("confidence", 0.0))

            if not ss:
                ss = "subject0"
            if not ts:
                ts = "target0"

            votes.append(
                {
                    "idx": i,
                    "temperature": temp,
                    "subject_slot": ss,
                    "target_slot": ts,
                    "alt_subject_slot": ass,
                    "alt_target_slot": ats,
                    "subject_reject_note": str(parsed.get("subject_reject_note", "") or "").strip(),
                    "target_reject_note": str(parsed.get("target_reject_note", "") or "").strip(),
                    "confidence": conf,
                    "error": err,
                    "raw": parsed,
                }
            )

        pair_counter = Counter((v["subject_slot"], v["target_slot"]) for v in votes)
        pair_conf = defaultdict(float)
        subj_counter = Counter(v["subject_slot"] for v in votes)
        tgt_counter = Counter(v["target_slot"] for v in votes)

        for v in votes:
            key = (v["subject_slot"], v["target_slot"])
            pair_conf[key] += 1.0 + 0.35 * v["confidence"]

        ranked_pairs = sorted(pair_conf.items(), key=lambda kv: (-kv[1], -pair_counter[kv[0]], kv[0][0], kv[0][1]))
        if ranked_pairs:
            best_pair = ranked_pairs[0][0]
            subject_slot, target_slot = best_pair
            top_score = ranked_pairs[0][1]
            total_score = sum(v for _, v in ranked_pairs)
            pair_top_conf = _clip01(top_score / total_score) if total_score > 0 else 0.0
        else:
            subject_slot, target_slot, pair_top_conf = "subject0", "target0", 0.0

        if subject_slot not in subject_slots:
            subject_slot = "subject0"
        if target_slot not in target_slots:
            target_slot = "target0"

        return {
            "votes": votes,
            "subject_slot": subject_slot,
            "target_slot": target_slot,
            "top_confidence": pair_top_conf,
            "subject_vote_counts": dict(subj_counter),
            "target_vote_counts": dict(tgt_counter),
            "pair_vote_counts": {f"{k[0]}|{k[1]}": v for k, v in pair_counter.items()},
        }, usage_total

    def _apply_prior_router(
        self,
        *,
        sample_id: str,
        mechanism: str,
        label: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        debug = {
            "enabled": not bool(self.cfg.disable_prior_router),
            "applied": False,
            "mechanism_rewritten": False,
            "label_rewritten": False,
            "notes": [],
            "thresholds": {
                "prior_mech_threshold": self.cfg.prior_mech_threshold,
                "prior_base_only_mech_threshold": self.cfg.prior_base_only_mech_threshold,
                "prior_label_threshold": self.cfg.prior_label_threshold,
            },
        }

        if self.cfg.disable_prior_router:
            debug["notes"].append("disabled")
            return mechanism, label, debug

        prior = self._prior_lookup(sample_id)
        base = prior.get("base", {})
        cot = prior.get("cot", {})
        debug["prior"] = {"base": base, "cot": cot}

        out_mech = mechanism
        out_label = label

        cot_mech = _canon_mech(cot.get("mechanism"))
        cot_mech_conf = _clip01(cot.get("mechanism_confidence"))
        base_mech = _canon_mech(base.get("mechanism"))
        base_mech_conf = _clip01(base.get("mechanism_confidence"))

        if cot_mech and cot_mech_conf >= float(self.cfg.prior_mech_threshold):
            out_mech = cot_mech
            debug["applied"] = True
            debug["mechanism_rewritten"] = (out_mech != mechanism)
            debug["notes"].append("mechanism_from_cot_prior")
        elif (not cot_mech) and base_mech and base_mech_conf >= float(self.cfg.prior_base_only_mech_threshold):
            out_mech = base_mech
            debug["applied"] = True
            debug["mechanism_rewritten"] = (out_mech != mechanism)
            debug["notes"].append("mechanism_from_base_prior")

        cot_label = _canon_label(cot.get("label"))
        cot_label_conf = _clip01(cot.get("label_confidence"))
        base_label = _canon_label(base.get("label"))
        base_label_conf = _clip01(base.get("label_confidence"))

        if cot_label and cot_label_conf >= float(self.cfg.prior_label_threshold):
            out_label = cot_label
            debug["applied"] = True
            debug["label_rewritten"] = (out_label != label)
            debug["notes"].append("label_from_cot_prior")
        elif base_label and base_label_conf >= float(self.cfg.prior_label_threshold):
            out_label = base_label
            debug["applied"] = True
            debug["label_rewritten"] = (out_label != label)
            debug["notes"].append("label_from_base_prior")

        if not debug["notes"]:
            debug["notes"].append("no_prior_trigger")
        return out_mech, out_label, debug

    def _stage_d_repair(
        self,
        *,
        common_payload: str,
        media_blocks: List[Dict[str, Any]],
        draft: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        parsed, usage, err = self._chat_json(
            system_prompt=p.STAGE_D_REPAIR_SYSTEM_PROMPT,
            user_text=p.build_stage_d_user_prompt(common_payload, draft),
            media_blocks=media_blocks,
            temperature=0.0,
        )

        mech = _canon_mech(parsed.get("mechanism")) or _canon_mech(draft.get("mechanism"))
        label = _canon_label(parsed.get("label")) or _canon_label(draft.get("label"))
        subject_slot = _slot_key(parsed.get("subject_slot"), "subject") or _slot_key(draft.get("subject_slot"), "subject")
        target_slot = _slot_key(parsed.get("target_slot"), "target") or _slot_key(draft.get("target_slot"), "target")

        if not mech:
            mech = "expressive_aggression"
        if not label:
            label = "provoke"
        if not subject_slot:
            subject_slot = "subject0"
        if not target_slot:
            target_slot = "target0"

        return {
            "mechanism": mech,
            "label": label,
            "subject_slot": subject_slot,
            "target_slot": target_slot,
            "repair_note": str(parsed.get("repair_note", err) or "").strip(),
            "raw": parsed,
            "error": err,
        }, usage

    def _stage_e_refine(
        self,
        *,
        common_payload: str,
        media_blocks: List[Dict[str, Any]],
        current_draft: Dict[str, Any],
        candidate_mechanisms: List[str],
        candidate_labels: List[str],
        stage_a_signal: Dict[str, Any],
        stage_b_summary: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        parsed, usage, err = self._chat_json(
            system_prompt=p.STAGE_E_HARD_REFINE_SYSTEM_PROMPT,
            user_text=p.build_stage_e_user_prompt(
                common_payload,
                current_draft=current_draft,
                candidate_mechanisms=candidate_mechanisms,
                candidate_labels=candidate_labels,
                stage_a_signal=stage_a_signal,
                stage_b_summary=stage_b_summary,
            ),
            media_blocks=media_blocks,
            temperature=0.0,
        )

        mech = _canon_mech(parsed.get("mechanism")) or _canon_mech(current_draft.get("mechanism"))
        label = _canon_label(parsed.get("label")) or _canon_label(current_draft.get("label"))

        if candidate_mechanisms and mech not in candidate_mechanisms:
            mech = candidate_mechanisms[0]
        if candidate_labels and label not in candidate_labels:
            label = candidate_labels[0]

        return {
            "mechanism": mech,
            "label": label,
            "refine_note": str(parsed.get("refine_note", err) or "").strip(),
            "raw": parsed,
            "error": err,
        }, usage

    def _slot_guard(
        self,
        *,
        text: str,
        audio_caption: str,
        subject_slot: str,
        target_slot: str,
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
    ) -> Tuple[str, str, Dict[str, Any]]:
        merged = f"{text or ''}\\n{audio_caption or ''}".strip()
        debug = {
            "subject_before": subject_slot,
            "target_before": target_slot,
            "subject_rewritten": False,
            "target_rewritten": False,
        }

        subject_scores = {
            s: _slot_evidence(merged, subject_slots.get(s, ""))
            for s in subject_slots.keys()
        }
        target_scores = {
            t: _slot_evidence(merged, target_slots.get(t, ""))
            for t in target_slots.keys()
        }

        best_subject = max(subject_scores.items(), key=lambda kv: kv[1])[0] if subject_scores else subject_slot
        best_target = max(target_scores.items(), key=lambda kv: kv[1])[0] if target_scores else target_slot

        if subject_scores and subject_scores.get(subject_slot, 0.0) < 0.12 and subject_scores.get(best_subject, 0.0) >= 0.28:
            subject_slot = best_subject
            debug["subject_rewritten"] = True

        if target_scores and target_scores.get(target_slot, 0.0) < 0.12 and target_scores.get(best_target, 0.0) >= 0.28:
            target_slot = best_target
            debug["target_rewritten"] = True

        debug["subject_after"] = subject_slot
        debug["target_after"] = target_slot
        debug["subject_scores"] = subject_scores
        debug["target_scores"] = target_scores
        return subject_slot, target_slot, debug

    def run(
        self,
        *,
        sample_id: str,
        text: str,
        audio_caption: str,
        subject_slots: Dict[str, str],
        target_slots: Dict[str, str],
        media_blocks: List[Dict[str, Any]],
        grounding_context: str = "",
    ) -> Dict[str, Any]:
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

        common_payload = p.build_common_user_payload(
            sample_id=str(sample_id),
            text=text,
            audio_caption=audio_caption,
            subject_slots=subject_slots,
            target_slots=target_slots,
        )
        if grounding_context:
            common_payload = common_payload + "\\n\\nGrounding context:\\n" + grounding_context

        stage_a, usage = self._stage_a(common_payload=common_payload, media_blocks=media_blocks)
        _sum_usage(total_usage, usage)

        stage_b, usage = self._stage_b(
            common_payload=common_payload,
            media_blocks=media_blocks,
            candidates=stage_a.get("top_candidates", []),
        )
        _sum_usage(total_usage, usage)

        stage_c, usage = self._stage_c(
            common_payload=common_payload,
            media_blocks=media_blocks,
            subject_slots=subject_slots,
            target_slots=target_slots,
        )
        _sum_usage(total_usage, usage)

        if stage_b.get("accepted", True):
            mech = _canon_mech(stage_b.get("winner_mechanism"))
            label = _canon_label(stage_b.get("winner_label"))
        else:
            top = stage_a.get("top_candidates", [])
            mech = _canon_mech(top[0]["mechanism"]) if top else ""
            label = _canon_label(top[0]["label"]) if top else ""

        if not mech:
            mech = "expressive_aggression"
        if not label:
            label = "provoke"

        mech, label, prior_router = self._apply_prior_router(
            sample_id=str(sample_id),
            mechanism=mech,
            label=label,
        )

        draft = {
            "mechanism": mech,
            "label": label,
            "subject_slot": stage_c.get("subject_slot", "subject0"),
            "target_slot": stage_c.get("target_slot", "target0"),
        }

        stage_d, usage = self._stage_d_repair(common_payload=common_payload, media_blocks=media_blocks, draft=draft)
        _sum_usage(total_usage, usage)

        should_refine = False
        top_conf = _clip01(stage_a.get("top_confidence", 0.0))
        if top_conf < float(self.cfg.hard_refine_top_conf_threshold):
            should_refine = True
        if not bool(stage_b.get("accepted", True)):
            should_refine = True

        signals = dict(stage_a.get("majority_signals", {}))
        if stage_d.get("label") == "mitigate" and (signals.get("ridicule_signal") or signals.get("reaction_bait_signal")):
            should_refine = True
        if stage_d.get("label") in {"mock", "provoke"} and signals.get("deescalation_signal") and top_conf < 0.85:
            should_refine = True

        stage_e = {
            "triggered": False,
            "mechanism": stage_d["mechanism"],
            "label": stage_d["label"],
            "refine_note": "skip",
        }

        if should_refine:
            mech_candidates = []
            label_candidates = []
            for c in stage_a.get("top_candidates", []):
                cm = _canon_mech(c.get("mechanism"))
                cl = _canon_label(c.get("label"))
                if cm and cm not in mech_candidates:
                    mech_candidates.append(cm)
                if cl and cl not in label_candidates:
                    label_candidates.append(cl)

            if stage_b.get("winner_mechanism"):
                wm = _canon_mech(stage_b.get("winner_mechanism"))
                if wm and wm not in mech_candidates:
                    mech_candidates.insert(0, wm)
            if stage_b.get("winner_label"):
                wl = _canon_label(stage_b.get("winner_label"))
                if wl and wl not in label_candidates:
                    label_candidates.insert(0, wl)

            refined, usage = self._stage_e_refine(
                common_payload=common_payload,
                media_blocks=media_blocks,
                current_draft=stage_d,
                candidate_mechanisms=mech_candidates,
                candidate_labels=label_candidates,
                stage_a_signal=signals,
                stage_b_summary=stage_b,
            )
            _sum_usage(total_usage, usage)
            stage_e = dict(refined)
            stage_e["triggered"] = True

        final_mech = _canon_mech(stage_e.get("mechanism")) or _canon_mech(stage_d.get("mechanism"))
        final_label = _canon_label(stage_e.get("label")) or _canon_label(stage_d.get("label"))

        if not final_mech:
            final_mech = "expressive_aggression"
        if not final_label:
            final_label = "provoke"

        subject_slot = _slot_key(stage_d.get("subject_slot"), "subject") or _slot_key(stage_c.get("subject_slot"), "subject")
        target_slot = _slot_key(stage_d.get("target_slot"), "target") or _slot_key(stage_c.get("target_slot"), "target")
        if not subject_slot:
            subject_slot = "subject0"
        if not target_slot:
            target_slot = "target0"

        subject_slot, target_slot, slot_guard = self._slot_guard(
            text=text,
            audio_caption=audio_caption,
            subject_slot=subject_slot,
            target_slot=target_slot,
            subject_slots=subject_slots,
            target_slots=target_slots,
        )

        if subject_slot not in subject_slots:
            subject_slot = "subject0"
        if target_slot not in target_slots:
            target_slot = "target0"

        return {
            "prediction": {
                "mechanism": MECH_DISPLAY.get(final_mech, "expressive aggression"),
                "label": final_label,
                "subject_slot": subject_slot,
                "target_slot": target_slot,
                "subject": subject_slots.get(subject_slot, ""),
                "target": target_slots.get(target_slot, ""),
            },
            "debug": {
                "stage_a": stage_a,
                "stage_b": stage_b,
                "stage_c": stage_c,
                "stage_d": stage_d,
                "stage_e": stage_e,
                "prior_router": prior_router,
                "slot_guard": slot_guard,
            },
            "token_usage": total_usage,
        }
