#!/usr/bin/env python
"""Auto-calibrate per-speaker, per-emotion emotion-control strength.

For every speaker the server exposes and every emotion, this sweeps the
emotion strength over a grid, generates a few neutral-content sentences, scores
each clip with the emotion2vec recogniser (``iic/emotion2vec_plus_large``), and
records the strength that maximises the mean probability of the *target*
emotion class.

The result is a ``calibration.json`` consumed by the server: at inference the
calibrated strength for (speaker, emotion) is folded into the emotion delta, so
the user's ``emotion_strength`` becomes a multiplier (default 1.0 = calibrated).

Run in a python env with ``funasr`` + ``librosa`` installed, against a running
server started with emotion directions:

    python scripts/calibrate_emotion_strength.py \
        --server http://localhost:1919 \
        --out emotion_directions/calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from collections import defaultdict

import librosa
import numpy as np

TARGET_CLASS = {
    "happy": "happy", "sad": "sad", "angry": "angry", "surprised": "surprised",
    "valence": "happy", "arousal": "surprised",
}
AXES = {"valence", "arousal"}
DEFAULT_SENTENCES = [
    "The meeting is scheduled for three o'clock on Tuesday afternoon.",
    "Please remember to bring the documents to the front desk.",
    "The train departs from platform nine in about ten minutes.",
]


def http_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def generate(server, spk, text, seed, extra, accurate=True):
    body = {"text": text, "seed": seed, "stream": True,
            "speaker_embedding_id": spk, "quality_enabled": False,
            "accurate_mode": accurate}
    body.update(extra)
    req = urllib.request.Request(server + "/tts/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    buf = bytearray()
    with urllib.request.urlopen(req, timeout=180) as r:
        while True:
            c = r.read(65536)
            if not c:
                break
            buf.extend(c)
    return np.frombuffer(bytes(buf), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:1919")
    ap.add_argument("--emotions", nargs="+", default=["happy", "sad", "angry", "surprised"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES)
    ap.add_argument("--min-prob", type=float, default=0.12,
                    help="A (speaker,emotion) counts toward the global default only above this.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sr-out", type=int, default=44100)
    ap.add_argument("--expressive", action="store_true",
                    help="Calibrate with accurate_mode=False (expressive token), which amplifies sliders.")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    caps = http_json(args.server + "/tts/capabilities")
    if not caps.get("emotion_enabled"):
        raise SystemExit("Server reports emotion control disabled (no directions loaded).")
    speakers = http_json(args.server + "/tts/speakers")["speakers"]
    print(f"Calibrating {len(speakers)} speakers x {len(args.emotions)} emotions "
          f"x {len(args.strengths)} strengths x {len(args.sentences)} sentences", flush=True)

    from funasr import AutoModel
    rec = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True, device="cpu")

    def target_prob(wav, tgt):
        y = librosa.resample(wav, orig_sr=args.sr_out, target_sr=16000) if args.sr_out != 16000 else wav
        r = rec.generate(y, granularity="utterance", extract_embedding=False)[0]
        labels = [str(l).split("/")[-1].strip().lower() for l in r["labels"]]
        return dict(zip(labels, (float(s) for s in r["scores"]))).get(tgt, 0.0)

    by_speaker = {}
    labels_map = {}
    achieved = defaultdict(list)  # emotion -> list[(strength, prob)] for the chosen cells
    for spk in speakers:
        sid, label = spk["id"], spk.get("label", spk["id"])
        labels_map[sid] = label
        by_speaker[sid] = {}
        for emo in args.emotions:
            tgt = TARGET_CLASS.get(emo, emo)
            best_s, best_p = args.strengths[0], -1.0
            curve = []
            for s in args.strengths:
                extra = ({"emotion_enabled": True, f"emotion_{emo}": 1.0, "emotion_strength": s}
                         if emo in AXES else
                         {"emotion_enabled": True, "emotion_sliders": {emo: 1.0}, "emotion_strength": s})
                probs = [target_prob(generate(args.server, sid, t, args.seed, extra,
                                              accurate=not args.expressive), tgt)
                         for t in args.sentences]
                mp = float(np.mean(probs))
                curve.append((s, mp))
                if mp > best_p:
                    best_p, best_s = mp, s
            by_speaker[sid][emo] = best_s
            if best_p >= args.min_prob:
                achieved[emo].append((best_s, best_p))
            print(f"  {label:16s} {emo:10s} -> strength {best_s:g} (prob {best_p:.2f})  "
                  f"curve={['%g:%.2f'%(s,p) for s,p in curve]}", flush=True)

    # Per-emotion default = median chosen strength among speakers where it worked;
    # else median over all speakers. Global default = median of those.
    default = {}
    for emo in args.emotions:
        worked = [s for s, _ in achieved.get(emo, [])]
        alls = [by_speaker[sid][emo] for sid in by_speaker]
        default[emo] = float(statistics.median(worked or alls))
    global_default = float(statistics.median(list(default.values()))) if default else 4.0

    out = {
        "objective": "emotion2vec_target_prob",
        "grid": args.strengths,
        "min_prob": args.min_prob,
        "global_default": global_default,
        "default": default,
        "by_speaker": by_speaker,
        "labels": labels_map,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote calibration -> {args.out}")
    print("default per emotion:", default, "global_default:", global_default)


if __name__ == "__main__":
    main()
