#!/usr/bin/env python
"""Evaluate emotion-control sliders with the emotion2vec recognizer.

Drives a running ZONOS2 server (``/tts/generate``) over a sweep of emotion
sliders and scores each rendered clip with ``iic/emotion2vec_plus_large`` via
FunASR. Reports, per (emotion, strength), the mean recognizer probability of the
*target* class and the predicted-label mix, so you can see whether turning a
slider up actually increases the intended emotion.

Run in a python env with ``funasr`` + ``librosa`` installed, pointed at a
server started with emotion directions:

    python scripts/eval_emotion_control.py \
        --server http://localhost:1919 \
        --emotions sad happy angry --strengths 0 3 6 9

The server does the GPU TTS work; emotion2vec runs on CPU here by default.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import Counter, defaultdict

import librosa
import numpy as np

# emotion2vec_plus_large emits bilingual labels like '生气/angry'; we key on the
# english side. Targets map a slider name to the recognizer class it should hit.
TARGET_CLASS = {
    "happy": "happy", "sad": "sad", "angry": "angry",
    "surprise": "surprised", "surprised": "surprised", "fear": "fearful",
    "valence": "happy", "arousal": "surprised",  # rough axis expectations
}

# Emotionally-neutral content so the recognizer scores delivery, not wording.
DEFAULT_SENTENCES = [
    "The meeting is scheduled for three o'clock on Tuesday afternoon.",
    "Please remember to bring the documents to the front desk.",
    "The train departs from platform nine in about ten minutes.",
]


def http_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def generate(server: str, speaker_id: str, text: str, seed: int, extra: dict) -> np.ndarray:
    body = {
        "text": text, "seed": seed, "stream": True,
        "speaker_embedding_id": speaker_id, "quality_enabled": False,
    }
    body.update(extra)
    req = urllib.request.Request(
        server + "/tts/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    buf = bytearray()
    with urllib.request.urlopen(req, timeout=180) as r:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
    return np.frombuffer(bytes(buf), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:1919")
    ap.add_argument("--speaker", default=None, help="speaker_embedding_id; default = first listed speaker")
    ap.add_argument("--emotions", nargs="+", default=["sad", "happy", "angry"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[0, 3, 6, 9])
    ap.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sr-out", type=int, default=44100, help="server PCM sample rate")
    ap.add_argument("--save-dir", default=None, help="optional dir to dump rendered wavs")
    args = ap.parse_args()

    speaker_id = args.speaker
    if speaker_id is None:
        speakers = http_json(args.server + "/tts/speakers")["speakers"]
        if not speakers:
            raise SystemExit("No speakers available on the server.")
        speaker_id = speakers[0]["id"]
        print(f"Using speaker: {speaker_id} ({speakers[0].get('label')})")

    caps = http_json(args.server + "/tts/capabilities")
    if not caps.get("emotion_enabled"):
        raise SystemExit("Server reports emotion control is disabled (no directions loaded).")
    print(f"Server emotion_names={caps.get('emotion_names')} axes={caps.get('emotion_axes')}")

    from funasr import AutoModel
    import soundfile as sf
    rec = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

    def score(wav: np.ndarray) -> dict:
        y = librosa.resample(wav, orig_sr=args.sr_out, target_sr=16000) if args.sr_out != 16000 else wav
        r = rec.generate(y, granularity="utterance", extract_embedding=False)[0]
        labels = [str(l).split("/")[-1].strip().lower() for l in r["labels"]]
        return dict(zip(labels, (float(s) for s in r["scores"])))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Baseline (emotion disabled) for reference.
    print("\n=== baseline (emotion disabled) ===")
    base_target = defaultdict(list)
    base_preds = Counter()
    for si, text in enumerate(args.sentences):
        wav = generate(args.server, speaker_id, text, args.seed, {"emotion_enabled": False})
        d = score(wav)
        base_preds[max(d, key=d.get)] += 1
        for emo in args.emotions:
            base_target[emo].append(d.get(TARGET_CLASS.get(emo, emo), 0.0))
    print("  predicted-label mix:", dict(base_preds))

    print("\n=== sweep: mean target-class probability over %d sentence(s) ===" % len(args.sentences))
    header = "emotion      target       " + "  ".join(f"s={s:g}" for s in args.strengths)
    print(header)
    print("-" * len(header))
    for emo in args.emotions:
        tgt = TARGET_CLASS.get(emo, emo)
        cells = []
        for s in args.strengths:
            probs, preds = [], Counter()
            for text in args.sentences:
                if s == 0:
                    extra = {"emotion_enabled": False}
                elif emo in ("valence", "arousal"):
                    extra = {"emotion_enabled": True, f"emotion_{emo}": 1.0, "emotion_strength": s}
                else:
                    extra = {"emotion_enabled": True, "emotion_sliders": {emo: 1.0}, "emotion_strength": s}
                wav = generate(args.server, speaker_id, text, args.seed, extra)
                if args.save_dir:
                    sf.write(os.path.join(args.save_dir, f"{emo}_s{s:g}_{len(probs)}.wav"), wav, args.sr_out)
                d = score(wav)
                probs.append(d.get(tgt, 0.0))
                preds[max(d, key=d.get)] += 1
            cells.append((float(np.mean(probs)), preds.most_common(1)[0][0]))
        row = "  ".join(f"{p:.2f}" for p, _ in cells)
        modes = ",".join(f"{m}" for _, m in cells)
        print(f"{emo:12s} {tgt:12s} {row}   | top-pred per step: {modes}")


if __name__ == "__main__":
    main()
