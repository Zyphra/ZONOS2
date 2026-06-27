#!/usr/bin/env python
"""Auto-calibrate per-emotion strength on a *cloned* speaker pool (audio refs).

Like ``calibrate_emotion_strength.py`` but instead of iterating the server's
cached default voices, it clones each reference wav on the fly via
``speaker_audio_base64``. Used to calibrate on an arbitrary held-out speaker
pool and produce an aggregated ``default`` per-emotion strength that can
transfer to unseen speakers.

IMPORTANT: run against a server whose emotion-directions dir has NO
calibration.json present (move it aside first), so ``emotion_strength=s`` means
the raw strength ``s`` and is not double-scaled by an existing calibration.

Run in a python env with ``funasr`` + ``librosa`` installed:

    python scripts/calibrate_emotion_strength_clone.py \
        --server http://localhost:1919 \
        --ref-dir /path/to/reference_wavs \
        --ref-ids 1 2 3 4 5 6 7 8 \
        --strengths 2 3 4 5 6 --expressive \
        --out emotion_directions/calibration.json
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import urllib.request
from collections import defaultdict

import librosa
import numpy as np

TARGET_CLASS = {"happy": "happy", "sad": "sad", "angry": "angry", "surprised": "surprised"}
DEFAULT_SENTENCES = [
    "The meeting is scheduled for three o'clock on Tuesday afternoon.",
    "Please remember to bring the documents to the front desk.",
]


def http_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def generate(server, ref_b64, name, text, seed, emo, strength, accurate):
    body = {"text": text, "seed": seed, "stream": True,
            "speaker_audio_base64": ref_b64, "speaker_audio_name": name,
            "quality_enabled": False, "accurate_mode": accurate,
            "emotion_enabled": True, "emotion_sliders": {emo: 1.0},
            "emotion_strength": strength}
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
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--ref-ids", nargs="+", type=int, required=True,
                    help="prompt_audio_<N>.wav indices to calibrate on")
    ap.add_argument("--emotions", nargs="+", default=["happy", "sad", "angry", "surprised"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[2, 3, 4, 5, 6])
    ap.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES)
    ap.add_argument("--min-prob", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sr-out", type=int, default=44100)
    ap.add_argument("--expressive", action="store_true")
    ap.add_argument("--out", default="calibration_clone.json")
    args = ap.parse_args()

    caps = http_json(args.server + "/tts/capabilities")
    if not caps.get("emotion_enabled"):
        raise SystemExit("Server reports emotion control disabled (no directions loaded).")
    if caps.get("emotion_calibrated"):
        print("WARNING: server has a calibration loaded; strengths will be double-scaled. "
              "Move calibration.json aside and restart for a clean sweep.", flush=True)

    refs = {}
    for i in args.ref_ids:
        with open(f"{args.ref_dir}/prompt_audio_{i}.wav", "rb") as f:
            refs[i] = base64.b64encode(f.read()).decode()
    print(f"Calibrating {len(refs)} cloned speakers x {len(args.emotions)} emotions "
          f"x {len(args.strengths)} strengths x {len(args.sentences)} sentences "
          f"(expressive={args.expressive})", flush=True)

    from funasr import AutoModel
    rec = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True, device="cpu")

    def target_prob(wav, tgt):
        if len(wav) < 1600:
            return 0.0
        y = librosa.resample(wav, orig_sr=args.sr_out, target_sr=16000) if args.sr_out != 16000 else wav
        r = rec.generate(y, granularity="utterance", extract_embedding=False)[0]
        labels = [str(l).split("/")[-1].strip().lower() for l in r["labels"]]
        return dict(zip(labels, (float(s) for s in r["scores"]))).get(tgt, 0.0)

    by_speaker = {}
    achieved = defaultdict(list)
    for i, b64 in refs.items():
        sid = f"ztts_prompt_{i}"
        by_speaker[sid] = {}
        for emo in args.emotions:
            tgt = TARGET_CLASS[emo]
            best_s, best_p, curve = args.strengths[0], -1.0, []
            for s in args.strengths:
                probs = [target_prob(generate(args.server, b64, f"prompt_{i}.wav", t,
                                              args.seed, emo, s, accurate=not args.expressive), tgt)
                         for t in args.sentences]
                mp = float(np.mean(probs))
                curve.append((s, mp))
                if mp > best_p:
                    best_p, best_s = mp, s
            by_speaker[sid][emo] = best_s
            if best_p >= args.min_prob:
                achieved[emo].append((best_s, best_p))
            print(f"  prompt_{i:<4} {emo:10s} -> strength {best_s:g} (prob {best_p:.2f})  "
                  f"curve={['%g:%.2f' % (s, p) for s, p in curve]}", flush=True)

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
        "calibrated_on": [f"prompt_{i}" for i in args.ref_ids],
        "expressive": args.expressive,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote calibration -> {args.out}")
    print("default per emotion:", default, "global_default:", global_default)
    print("n speakers where each emotion cleared min_prob:",
          {e: len(achieved.get(e, [])) for e in args.emotions})


if __name__ == "__main__":
    main()
