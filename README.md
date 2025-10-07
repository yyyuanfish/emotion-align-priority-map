# Emotion-Align Priority Map Design Draft

## 1. Project Overview
- **Goal:** Build a lightweight embodied agent that senses emotional cues from text, speech, and facial signals, then aligns its behavior and environment feedback in real time via a priority-map mechanism.
- **Core idea:** Reuse Jason Armitage’s **Plan → Localize → Policy** pipeline from navigation by mapping “routes/landmarks” to “affective plans/evidence.”
- **Scenario:** Small-scale embodied interaction in a lab setting (virtual room or street corner mock-up) focused on trust, synchrony, and immersion.

## 2. Key Research Questions and Hypothesis
1. Can a high-level Affective Plan improve temporal alignment between multimodal emotion signals and agent actions?
2. Does the approach outperform a strong late-fusion baseline on user experience metrics (emotional synchrony, trust, immersion)?
3. What benefits can small, in-domain pretraining deliver compared to large generic pretraining?

**Hypothesis:** A two-stage “plan → localize” pipeline (high-level social intent plus evidence boosting) will outperform single-stage fusion within a controllable, small-data setting.

## 3. System Architecture (Plan → Localize → Policy)
```
Inputs: text / speech / face
        ↓  (light encoders + normalization)
Affective Plan (temporal prior: social intent prediction)
        ↓
Cue Localization with Boost (cross-modal evidence selection + weighting)
        ↓
Controller / Policy (response style + environment feedback)
        ↓
Outputs: verbal response + environmental cue (light/sound)
```
- **Feedback loop:** Each output influences the next plan prediction.

## 4. Module Definitions
- **Affective Plan (temporal prior):** Predicts 3–5 high-level social intents (e.g., clarify, encourage, slow down, reassure) from recent dialogue history, providing a time-aware prior over relevant cues.
- **Cue Localization with Boost (feature-level):** Computes diagnostic weights for textual spans, acoustic segments, and facial frames given the plan, amplifying the most relevant evidence for the controller.
- **Controller / Policy:** Combines the plan and boosted features to choose response tone (neutral/supportive/enthusiastic), speaking tempo, and 2–3 environment adjustments (warm/cool lighting, calm/active ambience).

## 5. Data Resource Plan
- **Text emotion:** GoEmotions (fine-grained labels + valence/arousal mapping), EmpatheticDialogues (emotion-driven conversations with empathic responses), DailyDialog (emotion + dialogue acts).
- **Speech emotion:** IEMOCAP (rich improvised sessions), RAVDESS (controlled intensities, great for stress tests), CREMA-D (speaker diversity), ESD (Chinese/English; leverages CE-EmoTTS experience).
- **Facial expression:** AffectNet (large-scale valence/arousal), RAF-DB and FER+ (clean quick baselines), Aff-Wild2 (temporal video cues).
- **Multimodal benchmarks:** CMU-MOSI/MOSEI, CH-SIMS (Chinese sentiment), MELD (multi-party dialogue).
- **Embodied analogues:** Touchdown, RxR, R2R priority-map datasets to inspire affect “landmark” supervision.
- **In-domain micro-corpus:** Collect 200–400 short interactions with turn-level labels: user text/speech/face, chosen social intent, supporting evidence, response style, environment tweak.

## 6. Initial Plan Inventory Recommendation
1. **Clarify:** triggered by confusion or repeated questions.
2. **Encourage:** when the user sounds negative or hesitant.
3. **Slow Down:** when pace or affect becomes tense or rushed.
4. **Reassure:** in response to anxiety or frustration.
5. **Celebrate (reinforce positive)** (optional): when strong positive cues appear.

## 7. Resolving Conflicting Evidence
- Use the plan’s prior to normalize confidence weights across modalities.
- Add a hierarchical consistency check: if modalities strongly disagree, fall back to a “request clarification” safety strategy.
- Apply temporal smoothing (sliding window or dynamic time warping) to handle speech/face lagging behind text.

## 8. Low-Sample Pretraining Ideas
- **Plan prediction auxiliary task:** Predict the next social intent from short affect histories (contrastive + cross-entropy losses).
- **Evidence matching task:** Train a matcher on labeled key text/audio/face segments, borrowing VLN-style discriminative losses.
- **Cross-modal consistency verification:** Randomly perturb one modality and train the model to recover the correct evidence combination for robustness.

## 9. Prototype Scene Recommendation
- **Preferred start:** A small room with 2–3 controllable zones (lighting + ambience) for simpler capture of speech and facial data, better privacy, and stable acoustics.
- Street-corner or mixed-reality variants can follow once the room setup is validated.

## 10. Environment Feedback (non-intrusive)
- Lighting: warm white (relax/encourage), cool white (focus/clarify), soft dim light (reassure).
- Background audio: gentle white noise or nature sounds (calm), light upbeat tracks (encourage), near-silence (clarify/focus).
- Optional visual panel cues (color/icon changes) to aid interpretability.

## 11. Evaluation Plan
- **Offline metrics:** Valence/arousal concordance (CCC/Pearson), evidence selection accuracy (span/segment/frame), plan classification F1.
- **User study (small N):**
  - Participants: 12–18 (balanced demographics).
  - Tasks: 3–4 short scripted interactions (learning or consultation scenarios).
  - Measures: emotional synchrony (lagged correlation), short trust-in-automation scale, presence questionnaire, optional Godspeed.
  - Qualitative debrief: perceived understanding and timing alignment.
- **Ablations & stress tests:** Modality dropout (no speech/face), noise injection (fast speech, occlusions), disable Plan or Boost modules.
- **Success criterion:** ≥5–10% improvement over a strong late-fusion baseline on synchrony or trust; statistically significant offline evidence alignment gains (p < 0.05).

## 12. Ethics & Privacy Checklist
- Obtain written consent with clear data usage and retention policies.
- Store feature vectors instead of raw audio/video when possible; encrypt raw data if retention is required.
- Allow participants to withdraw and delete data; restrict access permissions.
- Avoid vulnerable populations; ensure equipment does not capture bystanders.
- Review environment cues for potential discomfort (e.g., flashing lights).

## 13. Risks and Mitigations
- **Subjective labels:** Limit to valence/arousal plus 6–8 discrete emotions; measure inter-rater agreement.
- **Data scarcity:** Leverage pretrained encoders and in-domain auxiliary tasks; start with text+speech before adding face.
- **Latency:** Fix processing windows, pre-extract facial features, and keep controller actions lightweight.

## 14. 12-Week Milestone Sketch
- **Weeks 1–2:** Finalize label schema, secure ethics approval, build annotation guidelines.
- **Weeks 3–4:** Set up prototype environment, clean public datasets, prototype encoders.
- **Weeks 5–6:** Implement Affective Plan and train the intent classifier.
- **Weeks 7–8:** Implement Cue Localization with Boost, finish evidence labeling/training.
- **Week 9:** Integrate controller and run baselines (late fusion, text-only, no-plan, no-boost).
- **Week 10:** Pilot study and UI/UX refinements.
- **Weeks 11–12:** Formal evaluation, stress tests, and final write-up.

## 15. Key Related Work (quick notes)
- **WACV 2023 Priority Map:** Two-stage planning/localization with low-sample in-domain pretraining; major gains on Touchdown.
- **Armitage PhD Thesis:** Views embodied intelligence as system-level alignment; advocates explicit mechanisms for unsynchronized sequences.
- **CE-EmoTTS, Whisper vs Humans, Fed Reports:** Provide reusable insights on cross-lingual zero-shot synthesis, speech rate/source effects, and event-window analytics for measurement design.

## 16. Open Questions
- Minimal definition of affective “landmarks”: combine valence change thresholds with archetypal phrases or expression templates.
- If trade-offs arise, prioritize **plan accuracy** (correct high-level intent) before refining evidence localization.
- Baseline goal: at least 105% of a strong text-only baseline and statistically significant gains on trust measures.
- Alignment pitfalls to pre-empt: modality latency, mismatched frame rates, lack of lag compensation—address via temporal buffers and explicit lag modeling.

## 17. Prototype Implementation Notes
- The `run_agent.py` harness now follows a **one-shot interaction flow**: a single user utterance (plus optional camera image path) triggers one agent reply and environment cue, matching the “one dialogue → one response” requirement.
- Controller replies are plan-aware: each response text references the chosen intent and the environment adjustments, mirroring the Plan → Localize → Policy logic described in this document.
- Emotion detection and plan recommendation can be delegated to an LLM via `utils/llm_stub.LLMEmotionAnalyzer`; the default configuration enables a stub provider that can be replaced with a real model while keeping the control flow unchanged.
- Telemetry (`telemetry.jsonl`) captures the full Plan → Localize → Policy trace per turn, supporting offline diagnostics even for single-shot runs.

