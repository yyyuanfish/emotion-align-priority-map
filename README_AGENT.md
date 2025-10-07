# Agent Mode (Embodied Emotion Alignment) — Minimal Implementation

This folder contains a minimal, test-friendly implementation of the Agent-mode
workflow described in the operational spec. It's intentionally lightweight and
uses randomized encoders so you can validate the control flow, telemetry, and
baseline switching quickly without heavy ML dependencies.

Files added:

- `agent_mode.py` — Core Emotion-Align agent with encoders, LLM-driven affective
  planning, localization, policy, act, telemetry, and history tracking.
- `run_agent.py` — Single-turn interactive runner: collects one user utterance
  (plus optional image path), executes the agent loop once, and prints plan,
  evidence, reply, and environment cues. Telemetry is written to
  `telemetry.jsonl`.
- `utils/llm_stub.py` — Pluggable LLM interface. The default stub mimics
  emotion analysis with a lexicon but can be replaced by a real model by
  overriding `_dispatch`.

Quick run (macOS / zsh):

```bash
python3 run_agent.py
```

Follow the prompts. The console reports the inferred intent (with source
`llm` if the analyzer produced it), evidence spans, the agent’s reply, and the
selected environment cues. Each run appends a record to `telemetry.jsonl`.

Notes:
- Swap out `utils/llm_stub.py` with your preferred provider to run real LLM
  analyses (e.g., OpenAI, Claude). The agent consumes the LLM’s emotion,
  intent, confidence, evidence spans, and suggested reply.
- Encoders remain lightweight placeholders; replace them with production-grade
  models as needed.
- Privacy: by default, `privacy.store_raw_av` is False and the runner expects
  optional image paths from disk rather than live capture.
