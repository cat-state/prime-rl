"""Diagnostic multiscale RGB error; not yet an accepted RL reward."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def error(a, b):
    values = []
    for size in (16, 32, 64):
        shape = (max(1, round(size * b.width / b.height)), size)
        x = np.asarray(a.resize(shape, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        y = np.asarray(b.resize(shape, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        values.append(float(np.abs(x - y).mean()))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("renders", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.dataset / "manifest.json").read_text())["records"]
    targets = {r["id"]: Image.open(r["image"]).convert("RGB") for r in rows}
    records = []
    for row in rows:
        path = args.renders / f"{row['id']}-seed-42.png"
        if not path.exists():
            continue
        target = targets[row["id"]]
        candidate = Image.open(path).convert("RGB")
        mean = tuple(np.asarray(target).mean(axis=(0, 1)).round().astype(int))
        records.append({
            "id": row["id"], "candidate_error": error(candidate, target),
            "identity_error": error(target, target),
            "target_mean_color_error": error(Image.new("RGB", target.size, mean), target),
            "black_error": error(Image.new("RGB", target.size), target),
            "shuffled_target_errors": {key: error(candidate, other) for key, other in targets.items() if key != row["id"]},
        })
    result = {"metric": "mean RGB absolute error at heights16,32,64, aspect-preserving resize, display RGB",
              "status": "diagnostic only; inspect degenerate baselines before reward adoption", "records": records}
    (args.renders / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
