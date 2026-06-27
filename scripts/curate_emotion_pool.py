#!/usr/bin/env python
"""Curate an emotion-labelled manifest from a raw speech pool using emotion2vec.

Scores every clip in a pool with ``iic/emotion2vec_plus_large`` and keeps only
clips the recognizer is confident about,
grouped by predicted emotion. Writes a manifest ``{emotion: [wavs]}`` that can
be fed to ``scripts/build_emotion_directions.py --manifest`` to build
recognizer-verified emotion directions -- crucially including a confident
``neutral`` set to anchor the directions.

Runs the recognizer on CPU by default (set --device cuda to use a GPU).

    python scripts/curate_emotion_pool.py \
        --pool /path/to/esd_wavs /path/to/more_wavs \
        --classes neutral happy sad angry surprised \
        --min-conf 0.6 --per-class-max 60 --out curated_emotion_manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import librosa
import numpy as np

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def find_wavs(patterns):
    out = []
    for pat in patterns:
        if os.path.isdir(pat):
            for ext in AUDIO_EXTS:
                out.extend(glob.glob(os.path.join(pat, "**", f"*{ext}"), recursive=True))
        else:
            out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", nargs="+", required=True, help="dirs or globs of audio")
    ap.add_argument("--classes", nargs="+",
                    default=["neutral", "happy", "sad", "angry", "surprised"])
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--per-class-max", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="curated_emotion_manifest.json")
    args = ap.parse_args()

    wavs = find_wavs(args.pool)
    print(f"Found {len(wavs)} clips in pool.", flush=True)
    if not wavs:
        raise SystemExit("Empty pool.")

    from funasr import AutoModel
    rec = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True, device=args.device)
    want = set(args.classes)

    # Collect (path, conf) per predicted class.
    pred_by_class = defaultdict(list)
    dist = defaultdict(int)
    for i, p in enumerate(wavs):
        try:
            y, _ = librosa.load(p, sr=16000)
            r = rec.generate(y, granularity="utterance", extract_embedding=False)[0]
            labels = [str(l).split("/")[-1].strip().lower() for l in r["labels"]]
            scores = [float(s) for s in r["scores"]]
            j = int(np.argmax(scores))
            label, conf = labels[j], scores[j]
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {p}: {exc}", flush=True)
            continue
        dist[label] += 1
        if label in want and conf >= args.min_conf:
            pred_by_class[label].append((p, conf))
        if (i + 1) % 50 == 0:
            print(f"  scored {i+1}/{len(wavs)}", flush=True)

    print("\nPredicted-label distribution over pool:")
    for k in sorted(dist, key=lambda k: -dist[k]):
        print(f"  {k:12s} {dist[k]}")

    # Keep the most-confident up to per-class-max per class.
    manifest = {}
    print("\nKept (conf >= %.2f, max %d/class):" % (args.min_conf, args.per_class_max))
    for cls in args.classes:
        items = sorted(pred_by_class.get(cls, []), key=lambda t: -t[1])[: args.per_class_max]
        if items:
            manifest[cls] = [p for p, _ in items]
        print(f"  {cls:12s} {len(items)}")

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {args.out}")
    if "neutral" not in manifest:
        print("WARNING: no confident 'neutral' clips kept; directions will lack a clean baseline.")


if __name__ == "__main__":
    main()
