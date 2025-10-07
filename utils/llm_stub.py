"""LLM-driven emotion and intent analyzer stub.

Provides an interface that can be backed by a real LLM. By default it runs a
lexicon-based heuristic so the system stays dependency-free, while exposing a
single `analyze` method that returns emotion, intent, and a response template.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


_NEGATIVE_LEXICON = {
    "en": ["sad", "upset", "angry", "depressed", "worried", "anxious", "confused", "lonely"],
    "zh": ["伤心", "难过", "生气", "害怕", "担心", "焦虑", "困惑", "孤独", "沮丧"],
}

_POSITIVE_LEXICON = {
    "en": ["happy", "glad", "excited", "joyful", "proud", "relieved"],
    "zh": ["开心", "高兴", "放松", "兴奋", "满意", "舒心"],
}

_AROUSAL_LEXICON = {
    "en": ["furious", "panic", "stressed", "energetic", "hyper"],
    "zh": ["激动", "紧张", "愤怒", "害怕", "亢奋"],
}

_INTENT_TEMPLATES = {
    "encourage": "I hear how hard this is. Let's keep moving one step at a time.",
    "slow_down": "I'll slow things down so we can process gently.",
    "reassure": "You're not alone—I’m staying close and attentive.",
    "celebrate": "That's wonderful progress. Let's keep that energy going.",
    "clarify": "Let me double-check so we stay aligned.",
}


@dataclass
class LLMResponse:
    emotion: str
    intent: str
    valence: float
    arousal: float
    confidence: float
    response: str
    evidence_spans: List[List[float]]
    rationale: str


class LLMEmotionAnalyzer:
    """Interface for LLM-based emotion understanding.

    The default provider ('stub') is lexicon-based; integrate a real LLM by
    subclassing and overriding `_dispatch`.
    """

    def __init__(self, cfg: Dict[str, any]):
        llm_cfg = cfg.get("llm", {})
        self.enabled: bool = llm_cfg.get("enabled", False)
        self.provider: str = llm_cfg.get("provider", "stub")
        self.model: str = llm_cfg.get("model", "emotion-align-stub")

    # Public API ---------------------------------------------------------
    def analyze(self, text: str, history: Optional[List[Dict]] = None) -> Optional[LLMResponse]:
        if not self.enabled:
            return None
        return self._dispatch(text, history or [])

    # Internal dispatch --------------------------------------------------
    def _dispatch(self, text: str, history: List[Dict]) -> LLMResponse:
        if self.provider == "stub":
            return self._stub_inference(text, history)
        raise NotImplementedError(
            f"LLM provider '{self.provider}' not implemented in this template. "
            "Subclass LLMEmotionAnalyzer and override `_dispatch` to plug in your model."
        )

    # Stub implementation ------------------------------------------------
    def _stub_inference(self, text: str, history: List[Dict]) -> LLMResponse:
        tokens_lower = text.lower()
        valence = 0.0
        arousal = 0.2
        spans: List[List[float]] = []

        def mark_span(word: str, weight: float = 0.6):
            idx = text.find(word)
            if idx >= 0:
                spans.append([idx, idx + len(word), weight])

        matched_neg = any(word in tokens_lower for word in _NEGATIVE_LEXICON["en"]) or any(
            word in text for word in _NEGATIVE_LEXICON["zh"]
        )
        matched_pos = any(word in tokens_lower for word in _POSITIVE_LEXICON["en"]) or any(
            word in text for word in _POSITIVE_LEXICON["zh"]
        )

        if matched_neg:
            valence -= 0.65
            for word in _NEGATIVE_LEXICON["en"]:
                if word in tokens_lower:
                    mark_span(word, 0.75)
            for word in _NEGATIVE_LEXICON["zh"]:
                if word in text:
                    mark_span(word, 0.85)
        if matched_pos:
            valence += 0.6
            for word in _POSITIVE_LEXICON["en"]:
                if word in tokens_lower:
                    mark_span(word, 0.7)
            for word in _POSITIVE_LEXICON["zh"]:
                if word in text:
                    mark_span(word, 0.8)

        if any(word in tokens_lower for word in _AROUSAL_LEXICON["en"]) or any(
            word in text for word in _AROUSAL_LEXICON["zh"]
        ):
            arousal += 0.45

        # Light memory effect
        if history:
            past_valences = [entry.get("plan_valence", 0.0) for entry in history if "plan_valence" in entry]
            if past_valences:
                valence = 0.7 * valence + 0.3 * (sum(past_valences) / len(past_valences))

        valence = max(-1.0, min(1.0, valence + random.uniform(-0.05, 0.05)))
        arousal = max(-1.0, min(1.0, arousal + abs(valence) * 0.4))

        if valence <= -0.35:
            intent = "encourage"
            emotion = "sadness"
        elif arousal >= 0.65 and valence < 0.1:
            intent = "slow_down"
            emotion = "anxiety"
        elif valence >= 0.4:
            intent = "celebrate"
            emotion = "joy"
        elif valence <= -0.1:
            intent = "clarify"
            emotion = "confusion"
        else:
            intent = "reassure"
            emotion = "neutral"

        response = _INTENT_TEMPLATES.get(intent, _INTENT_TEMPLATES["clarify"])
        confidence = min(0.95, 0.6 + abs(valence) * 0.3 + random.uniform(0.0, 0.05))
        rationale = f"Stub LLM inferred emotion '{emotion}' with intent '{intent}' from text patterns."
        if not spans:
            spans = [[0, min(len(text), 30), 0.4]]

        return LLMResponse(
            emotion=emotion,
            intent=intent,
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            response=response,
            evidence_spans=spans,
            rationale=rationale,
        )
