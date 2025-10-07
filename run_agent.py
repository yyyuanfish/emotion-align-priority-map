"""Single-turn runner for the Emotion-Align agent.

Prompts the user for one utterance (and optional camera image) and prints
the agent's reply plus environment cues, then exits. Designed for quick
one-shot testing where each dialogue round yields exactly one response.
"""

from __future__ import annotations

import pathlib
from typing import Optional

from agent_mode import make_agent


def _load_image_bytes(path: str) -> Optional[bytes]:
    if not path:
        return None
    img_path = pathlib.Path(path).expanduser()
    if not img_path.exists() or not img_path.is_file():
        print(f"[warn] Image path '{img_path}' not found. Continuing without face input.")
        return None
    try:
        return img_path.read_bytes()
    except Exception as exc:
        print(f"[warn] Failed to read image '{img_path}': {exc}. Continuing without face input.")
        return None


def main():
    agent = make_agent()
    print("Emotion-Align one-shot demo")
    print("Enter your message; optionally supply a camera image path to accompany it.\n")

    try:
        user_text = input("Your message: ").strip()
        if not user_text:
            print("No message provided. Exiting without response.")
            return

        img_path = input("Optional image path (press Enter to skip): ").strip()
        face_payload = _load_image_bytes(img_path)

        result = agent.run_turn(1, text_t=user_text, face_t=face_payload)
        telemetry = result["telemetry"]
        reply = result["action"]["reply"]["text"]
        style = result["action"]["reply"]["style"]
        env = result["action"]["env"]
        intent = telemetry["plan"]["intent"]
        evidence = telemetry["evidence"]
        plan_source = telemetry["plan"].get("source", "heuristic")
        plan_emotion = telemetry["plan"].get("emotion")
        print(f"\n[Plan] intent={intent} confidence={telemetry['plan']['confidence']:.2f} source={plan_source} emotion={plan_emotion}")
        print(f"[Evidence] text_spans={evidence.get('text_spans')} speech_segments={evidence.get('speech_segments')} face_frames={evidence.get('face_frames')}")
        print(f"Agent ({style}): {reply}")
        print(f"Environment cue: {env}")
    finally:
        agent.close()
        print("\nSession ended.")


if __name__ == "__main__":
    main()
