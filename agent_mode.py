"""Agent-mode workflow implementation for the Emotion-Align prototype.

Implements the Plan → Localize → Policy loop described in the design draft
with lightweight, test-friendly components. The encoders are intentionally
simple/random to keep the harness dependency-free, while the downstream
modules follow the structure of the proposed system (affective plan, cue
localization with boost, conflict handling, controller/policy, telemetry).
"""

from __future__ import annotations

import json
import random
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.llm_stub import LLMEmotionAnalyzer, LLMResponse


DEFAULT_CONFIG: Dict[str, Any] = {
    "agent": {
        "mode": "full",  # {full, text_only, no_plan, no_boost}
        "turn_horizon": 4,
        "max_latency_ms": 250,
    },
    "labels": {
        "emotions": ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"],
        "valence_range": [-1.0, 1.0],
        "arousal_range": [-1.0, 1.0],
        "intent_inventory": ["clarify", "encourage", "slow_down", "reassure", "celebrate"],
    },
    "plan": {
        "neg_threshold": -0.25,
        "low_threshold": -0.05,
        "high_threshold": 0.35,
        "arousal_fast_threshold": 0.55,
        "history_weight": 0.35,
        "min_confidence": 0.55,
    },
    "localize": {
        "top_k_spans": 2,
        "top_k_segments": 2,
        "top_k_frames": 2,
        "boost_gain": 1.5,
        "conflict_delta": 0.6,
        "lag_tolerance": 2,
    },
    "policy": {
        "styles": ["neutral", "supportive", "enthusiastic", "calming"],
        "env_controls": {
            "light_hue": ["cool", "warm"],
            "light_level": ["low", "mid", "high"],
            "sound": ["off", "calm", "bright"],
        },
        "safety_reply": "I noticed the signals disagree. Could you clarify how you're feeling?",
    },
    "privacy": {"store_raw_av": False},
    "llm": {
        "enabled": True,
        "provider": "stub",  # swap to custom provider when integrating a real LLM
        "model": "emotion-align-stub",
    },
}


# ---------------------------------------------------------------------------
# Encoder stubs (randomised for harness testing)
# ---------------------------------------------------------------------------


def _bounded_rand(base: float, spread: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, base + random.uniform(-spread, spread)))


def encode_text(s: str, emotions: Iterable[str]) -> Dict[str, Any]:
    """Very small heuristic encoder that mimics sentiment extraction."""
    base = 0.0
    low = s.lower()
    positive_en = {"great", "good", "love", "awesome", "calmer", "thanks", "happy"}
    negative_en = {"sad", "sorry", "unhappy", "bad", "confused", "angry", "upset", "depressed"}
    high_arousal_terms_en = {"excited", "furious", "panic", "rush", "fast"}

    # basic Chinese lexicon to support multilingual demos
    positive_zh = {"开心", "高兴", "放心", "不错", "喜欢", "安心", "谢谢"}
    negative_zh = {"伤心", "难过", "生气", "害怕", "紧张", "担心", "沮丧", "崩溃", "痛苦"}
    high_arousal_terms_zh = {"紧张", "激动", "愤怒", "害怕"}

    spans: List[List[float]] = []

    def _mark_span(word: str, weight: float = 0.6) -> None:
        idx = s.find(word)
        if idx >= 0:
            spans.append([idx, idx + len(word), weight])

    if any(w in low for w in positive_en) or any(w in s for w in positive_zh):
        base += 0.6
        for w in positive_en:
            if w in low:
                _mark_span(w, 0.65)
        for w in positive_zh:
            if w in s:
                _mark_span(w, 0.7)

    if any(w in low for w in negative_en) or any(w in s for w in negative_zh):
        base -= 0.6
        for w in negative_en:
            if w in low:
                _mark_span(w, 0.7)
        for w in negative_zh:
            if w in s:
                _mark_span(w, 0.8)

    arousal_boost = 0.0
    if any(w in low for w in high_arousal_terms_en) or any(w in s for w in high_arousal_terms_zh):
        arousal_boost = 0.4
        for w in high_arousal_terms_en:
            if w in low:
                _mark_span(w, 0.7)
        for w in high_arousal_terms_zh:
            if w in s:
                _mark_span(w, 0.75)

    arousal = abs(base) * 0.4 + max(arousal_boost, 0.2)
    valence = _bounded_rand(base, 0.15)
    arousal = _bounded_rand(arousal, 0.2)
    emo_logits = {e: random.random() for e in emotions}
    if not spans:
        # fallback span covering the utterance for transparency
        spans.append([0, min(len(s), 30), _bounded_rand(abs(valence), 0.2, 0.25, 0.6)])
    emb = [random.random() for _ in range(16)]
    return {"valence": valence, "arousal": arousal, "emo_logits": emo_logits, "spans": spans, "emb": emb}


def encode_speech(wav: Optional[Any], emotions: Iterable[str]) -> Optional[Dict[str, Any]]:
    if wav is None:
        return None
    prosody = {
        "f0": random.uniform(80, 220),
        "energy": random.uniform(0.1, 1.0),
        "rate": random.uniform(2.5, 6.0),
    }
    valence = random.uniform(-0.55, 0.55)
    arousal = random.uniform(-0.2, 0.9)
    emo_logits = {e: random.random() for e in emotions}
    segments = [[0.0, 1.0, _bounded_rand(abs(valence), 0.2, 0.2, 1.0)]]
    emb = [random.random() for _ in range(16)]
    return {"prosody": prosody, "valence": valence, "arousal": arousal, "emo_logits": emo_logits, "segments": segments, "emb": emb}


def encode_face(frame: Optional[Any], emotions: Iterable[str]) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    aus = {"AU1": random.random(), "AU4": random.random(), "AU12": random.random()}
    valence = random.uniform(-0.4, 0.7)
    arousal = random.uniform(-0.3, 0.8)
    emo_logits = {e: random.random() for e in emotions}
    frames = [[0, _bounded_rand(abs(valence), 0.3, 0.2, 1.0)]]
    emb = [random.random() for _ in range(16)]
    return {"aus": aus, "valence": valence, "arousal": arousal, "emo_logits": emo_logits, "frames": frames, "emb": emb}


# ---------------------------------------------------------------------------
# Plan → Localize → Policy modules
# ---------------------------------------------------------------------------


def _safe_mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


@dataclass
class PlanOutput:
    intent: str
    confidence: float
    mean_valence: float
    mean_arousal: float
    rationale: Dict[str, Any]
    source: str = "heuristic"
    emotion: Optional[str] = None


class AffectivePlanModule:
    """Predicts the high-level social intent given recent history and features."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.intent_inventory = cfg["labels"]["intent_inventory"]
        self.history_buffer: List[Dict[str, float]] = []

    def _record_history(self, valence: float, arousal: float) -> None:
        self.history_buffer.append({"valence": valence, "arousal": arousal})
        if len(self.history_buffer) > 32:
            self.history_buffer = self.history_buffer[-32:]

    def _summarise_modalities(self, features: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[float, float]:
        vals = [f["valence"] for f in features.values() if f and "valence" in f]
        ars = [f["arousal"] for f in features.values() if f and "arousal" in f]
        return _safe_mean(vals), _safe_mean(ars)

    def _blend_with_history(self, valence: float, arousal: float) -> Tuple[float, float]:
        if not self.history_buffer:
            return valence, arousal
        horizon = min(len(self.history_buffer), 5)
        past_vals = [entry["valence"] for entry in self.history_buffer][-horizon:]
        past_ars = [entry["arousal"] for entry in self.history_buffer][-horizon:]
        weight = self.cfg["plan"]["history_weight"]
        blended_val = (1 - weight) * valence + weight * _safe_mean(past_vals)
        blended_ar = (1 - weight) * arousal + weight * _safe_mean(past_ars)
        return blended_val, blended_ar

    def infer_intent(
        self,
        features: Dict[str, Optional[Dict[str, Any]]],
        llm: Optional[LLMResponse] = None,
    ) -> PlanOutput:
        if llm:
            intent = llm.intent if llm.intent in self.intent_inventory else "clarify"
            confidence = max(self.cfg["plan"]["min_confidence"], min(0.99, llm.confidence))
            self._record_history(llm.valence, llm.arousal)
            return PlanOutput(
                intent=intent,
                confidence=confidence,
                mean_valence=llm.valence,
                mean_arousal=llm.arousal,
                rationale={
                    "source": "llm",
                    "emotion": llm.emotion,
                    "llm_model": self.cfg["llm"].get("model"),
                    "llm_rationale": llm.rationale,
                },
                source="llm",
                emotion=llm.emotion,
            )

        mean_val, mean_ar = self._summarise_modalities(features)
        blended_val, blended_ar = self._blend_with_history(mean_val, mean_ar)
        thresholds = self.cfg["plan"]

        if blended_val <= thresholds["neg_threshold"]:
            intent = "encourage"
            confidence = 0.75
        elif blended_ar >= thresholds["arousal_fast_threshold"] and blended_val <= thresholds["low_threshold"]:
            intent = "slow_down"
            confidence = 0.7
        elif blended_val >= thresholds["high_threshold"]:
            intent = "celebrate"
            confidence = 0.72
        elif blended_val <= thresholds["low_threshold"]:
            intent = "clarify"
            confidence = 0.6
        else:
            intent = "reassure"
            confidence = 0.65

        confidence = max(thresholds["min_confidence"], min(0.95, confidence + random.uniform(-0.05, 0.05)))
        self._record_history(blended_val, blended_ar)
        return PlanOutput(
            intent=intent,
            confidence=confidence,
            mean_valence=blended_val,
            mean_arousal=blended_ar,
            rationale={
                "raw_valence": mean_val,
                "raw_arousal": mean_ar,
                "history_size": len(self.history_buffer),
            },
            source="heuristic",
        )


@dataclass
class LocalizationFlags:
    needs_clarification: bool = False
    modality_drop: Dict[str, bool] = None
    conflict_score: float = 0.0


class CueLocalizer:
    """Selects evidence spans/segments/frames and boosts them for the policy."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _slice_topk(self, values: List[List[float]], k: int) -> List[List[float]]:
        return values[:k] if values else []

    def _detect_conflict(self, features: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[bool, float]:
        vals = [f["valence"] for f in features.values() if f and "valence" in f]
        if len(vals) < 2:
            return False, 0.0
        max_val = max(vals)
        min_val = min(vals)
        delta = abs(max_val - min_val)
        needs_clarification = delta >= self.cfg["localize"]["conflict_delta"]
        return needs_clarification, delta

    def localize(self, plan: PlanOutput, features: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[Dict[str, Any], LocalizationFlags]:
        txt = features.get("txt")
        spe = features.get("spe")
        face = features.get("face")
        conf, delta = self._detect_conflict({"txt": txt, "spe": spe, "face": face})

        text_spans = self._slice_topk(txt.get("spans", []) if txt else [], self.cfg["localize"]["top_k_spans"])
        speech_segments = self._slice_topk(spe.get("segments", []) if spe else [], self.cfg["localize"]["top_k_segments"])
        face_frames = self._slice_topk(face.get("frames", []) if face else [], self.cfg["localize"]["top_k_frames"])

        evidence = {
            "text_spans": text_spans,
            "speech_segments": speech_segments,
            "face_frames": [[int(idx), float(score)] for idx, score in face_frames],
        }

        flags = LocalizationFlags(
            needs_clarification=conf,
            modality_drop={"text": txt is None, "speech": spe is None, "face": face is None},
            conflict_score=delta,
        )
        return evidence, flags

    def apply_boost(self, evidence: Dict[str, Any], gain: float) -> Dict[str, Any]:
        boosted = {}
        for key, items in evidence.items():
            boosted_items: List[List[float]] = []
            for item in items:
                if len(item) >= 3:
                    boosted_items.append([item[0], item[1], item[2] * gain])
                elif len(item) == 2:
                    boosted_items.append([item[0], item[1] * gain])
                else:
                    boosted_items.append(item)
            boosted[key] = boosted_items
        return boosted


class ControllerPolicy:
    """Maps plan + boosted evidence to a response style and environment tweak."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _default_env(self) -> Dict[str, str]:
        return {"light_hue": "warm", "light_level": "mid", "sound": "calm"}

    def _format_environment_summary(self, env: Dict[str, str]) -> str:
        return f"(env → light: {env['light_hue']} / {env['light_level']}, sound: {env['sound']})"

    def _compose_reply(self, intent: str, style: str, env: Dict[str, str], flags: LocalizationFlags) -> str:
        env_summary = self._format_environment_summary(env)
        if flags.needs_clarification:
            return f"I'm sensing mixed signals, so I'm pausing to ask for clarity. {env_summary}"

        if intent == "encourage":
            return f"I'll stay supportive and help you keep momentum. {env_summary}"
        if intent == "slow_down":
            return f"Let's ease the pace together—focusing on calm, steady steps. {env_summary}"
        if intent == "reassure":
            return f"I'm here with you and we'll work through this carefully. {env_summary}"
        if intent == "celebrate":
            return f"That positive shift is great! I'll keep the energy bright. {env_summary}"
        return f"I'll clarify with you so we stay aligned. {env_summary}"

    def select_action(
        self,
        plan: PlanOutput,
        evidence: Dict[str, Any],
        flags: LocalizationFlags,
        context: Dict[str, Any],
        llm: Optional[LLMResponse] = None,
    ) -> Dict[str, Any]:
        env = self._default_env()
        style = "neutral"
        text = "Thanks — tell me more when you're ready."

        if flags.needs_clarification:
            style = "neutral"
            env["light_hue"] = "cool"
            env["sound"] = "off"
            text = self._compose_reply(plan.intent, style, env, flags)
        else:
            if plan.intent == "encourage":
                style = "supportive"
                env["light_hue"] = "warm"
                env["light_level"] = "mid"
                env["sound"] = "calm"
                text = self._compose_reply(plan.intent, style, env, flags)
            elif plan.intent == "slow_down":
                style = "calming"
                env["sound"] = "off"
                env["light_level"] = "low"
                text = self._compose_reply(plan.intent, style, env, flags)
            elif plan.intent == "reassure":
                style = "supportive"
                env["light_level"] = "low"
                text = self._compose_reply(plan.intent, style, env, flags)
            elif plan.intent == "celebrate":
                style = "enthusiastic"
                env["light_level"] = "high"
                env["sound"] = "bright"
                text = self._compose_reply(plan.intent, style, env, flags)
            else:  # clarify or fallback
                style = "neutral"
                env["light_hue"] = "cool"
                text = self._compose_reply(plan.intent, style, env, flags)

        if context.get("override_env"):
            env.update(context["override_env"])
            text = self._compose_reply(plan.intent, style, env, flags)
        if llm and not flags.needs_clarification:
            text = f"{llm.response} {self._format_environment_summary(env)}"
        return {"reply": {"text": text, "style": style}, "env": env, "intent": plan.intent}


# ---------------------------------------------------------------------------
# Agent harness
# ---------------------------------------------------------------------------


class AgentMode:
    """High-level orchestrator for the Emotion-Align prototype agent."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, telemetry_path: str = "telemetry.jsonl"):
        self.cfg = deepcopy(DEFAULT_CONFIG)
        if cfg:
            # shallow merge for top-level keys; nested dicts can be overridden as needed
            for key, value in cfg.items():
                if isinstance(value, dict) and key in self.cfg:
                    self.cfg[key].update(value)
                else:
                    self.cfg[key] = value

        self.telemetry_path = telemetry_path
        self.history: List[Dict[str, Any]] = []
        self.llm = LLMEmotionAnalyzer(self.cfg)
        self.plan_module = AffectivePlanModule(self.cfg)
        self.localizer = CueLocalizer(self.cfg)
        self.controller = ControllerPolicy(self.cfg)
        self._tele = open(self.telemetry_path, "a", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Lifecycle helpers                                                  #
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self._tele.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Perception subroutines                                             #
    # ------------------------------------------------------------------ #
    def _perceive(
        self,
        text_t: str,
        speech_t: Optional[Any],
        face_t: Optional[Any],
    ) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Optional[LLMResponse]]:
        emotions = self.cfg["labels"]["emotions"]
        F_txt = encode_text(text_t, emotions)
        F_spe = encode_speech(speech_t, emotions) if speech_t is not None else None
        F_face = encode_face(face_t, emotions) if face_t is not None else None
        llm_info = self.llm.analyze(text_t, self.history)
        if llm_info:
            F_txt["valence"] = llm_info.valence
            F_txt["arousal"] = llm_info.arousal
            F_txt["spans"] = llm_info.evidence_spans
            F_txt["llm_intent"] = llm_info.intent
            F_txt["llm_emotion"] = llm_info.emotion
            F_txt["llm_confidence"] = llm_info.confidence
            F_txt["llm_response"] = llm_info.response
            F_txt["llm_rationale"] = llm_info.rationale
        return {"txt": F_txt, "spe": F_spe, "face": F_face}, llm_info

    # ------------------------------------------------------------------ #
    # Telemetry helpers                                                  #
    # ------------------------------------------------------------------ #
    def _log_telemetry(self, entry: Dict[str, Any]) -> None:
        self._tele.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._tele.flush()

    def _update_history(self, record: Dict[str, Any]) -> None:
        self.history.append(record)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def run_turn(
        self,
        t: int,
        text_t: str,
        speech_t: Optional[Any] = None,
        face_t: Optional[Any] = None,
        ctx_t: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx_t = ctx_t or {}
        mode = self.cfg["agent"]["mode"]
        start_time = time.time()

        # 1) Perception
        features, llm_info = self._perceive(text_t, speech_t, face_t)
        if mode == "text_only":
            features["spe"], features["face"] = None, None

        # 2) Plan
        if mode == "no_plan":
            plan = PlanOutput(intent="clarify", confidence=1.0, mean_valence=0.0, mean_arousal=0.0, rationale={"bypass": True})
        else:
            plan = self.plan_module.infer_intent(features, llm_info)

        # 3) Localization
        evidence, flags = self.localizer.localize(plan, features)
        if mode != "no_boost":
            evidence = self.localizer.apply_boost(evidence, self.cfg["localize"]["boost_gain"])

        # 4) Policy / Act
        action = self.controller.select_action(plan, evidence, flags, ctx_t, llm=llm_info)
        effects = self._act(action)

        # 5) Telemetry & history
        latency_ms = int((time.time() - start_time) * 1000) + effects.get("latency_ms", 0)
        tele_entry = {
            "t": t,
            "mode": mode,
            "plan": {
                "intent": plan.intent,
                "confidence": plan.confidence,
                "mean_valence": plan.mean_valence,
                "mean_arousal": plan.mean_arousal,
                "rationale": plan.rationale,
                "source": plan.source,
                "emotion": plan.emotion,
                "llm_enabled": bool(llm_info),
            },
            "evidence": evidence,
            "flags": {
                "needs_clarification": flags.needs_clarification,
                "conflict_score": flags.conflict_score,
                "modality_drop": flags.modality_drop,
            },
            "policy": {
                "reply_style": action["reply"]["style"],
                "reply_text": action["reply"]["text"],
                "env": action["env"],
            },
            "latency_ms": latency_ms,
        }
        self._log_telemetry(tele_entry)

        history_record = {
            "turn": t,
            "inputs": {"text": text_t, "speech": bool(speech_t), "face": bool(face_t)},
            "plan": plan.intent,
            "action": action,
            "flags": {
                "needs_clarification": flags.needs_clarification,
                "conflict_score": flags.conflict_score,
            },
            "effects": effects,
            "plan_valence": plan.mean_valence,
            "plan_arousal": plan.mean_arousal,
        }
        self._update_history(history_record)

        return {"telemetry": tele_entry, "action": action, "effects": effects}

    # ------------------------------------------------------------------ #
    # Acting                                                             #
    # ------------------------------------------------------------------ #
    def _act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        time.sleep(random.uniform(0.02, 0.08))
        latency_ms = int((time.time() - start) * 1000)
        ok = latency_ms <= self.cfg["agent"]["max_latency_ms"]
        return {"ok": ok, "latency_ms": latency_ms}


def make_agent(cfg: Optional[Dict[str, Any]] = None, telemetry_path: str = "telemetry.jsonl") -> AgentMode:
    return AgentMode(cfg=cfg, telemetry_path=telemetry_path)
