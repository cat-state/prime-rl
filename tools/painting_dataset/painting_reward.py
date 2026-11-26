"""Diagnostic spatial color/edge reward; requires calibration before RL use."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def score(target_path, rendered_path):
    with Image.open(target_path) as image:
        target = image.convert("RGB")
    with Image.open(rendered_path) as image:
        candidate = image.convert("RGB")
    if abs(candidate.width / candidate.height - target.width / target.height) > 0.02:
        raise ValueError("Render must preserve target aspect ratio")
    color, edge = [], []
    for height in (16, 32, 64, 128):
        size = (max(2, round(height * target.width / target.height)), height)
        a = np.asarray(target.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        b = np.asarray(candidate.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        color.append(float(np.abs(a - b).mean()))
        errors = []
        for axis in (0, 1):
            da, db = np.diff(a, axis=axis), np.diff(b, axis=axis)
            errors.append(float(np.abs(da - db).mean() / (0.02 + np.abs(da).mean())))
        edge.append(float(np.mean(errors)))
    color_error, edge_error = float(np.mean(color)), float(np.mean(edge))
    return {"reward": float(np.exp(-4 * color_error - edge_error)),
            "color_error": color_error, "edge_error": edge_error,
            "metric_version": "spatial-color-edge-v1-diagnostic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [r for r in json.loads(args.manifest.read_text())["records"] if r.get("image")][:args.limit]
    results = []
    for index, row in enumerate(records):
        target = Image.open(row["image"]).convert("RGB")
        target.thumbnail((256, 256))
        folder = args.output / str(index)
        folder.mkdir()
        reference = folder / "target.png"
        target.save(reference)
        pixels = np.asarray(target)
        mean = tuple(np.rint(pixels.mean(axis=(0, 1))).astype(int))
        wrong = Image.open(records[(index + 1) % len(records)]["image"]).convert("RGB").resize(target.size)
        rng = np.random.default_rng(42)
        variants = {"identity": target, "blur2": target.filter(ImageFilter.GaussianBlur(2)),
                    "blur8": target.filter(ImageFilter.GaussianBlur(8)),
                    "mean_color": Image.new("RGB", target.size, mean),
                    "black": Image.new("RGB", target.size), "wrong_artwork": wrong,
                    "shuffled_pixels": Image.fromarray(rng.permutation(pixels.reshape(-1, 3)).reshape(pixels.shape))}
        metrics = {}
        for name, variant in variants.items():
            path = folder / f"{name}.png"
            variant.save(path)
            metrics[name] = score(reference, path)
        results.append({"id": row["id"], "metrics": metrics})
    summary = {"status": "diagnostic; generated-candidate and visual ranking checks still required",
               "records": results,
               "identity_blur_order_passes": sum(r["metrics"]["identity"]["reward"] >
                                                r["metrics"]["blur2"]["reward"] >
                                                r["metrics"]["blur8"]["reward"] for r in results),
               "blur8_beats_mean_passes": sum(r["metrics"]["blur8"]["reward"] >
                                              r["metrics"]["mean_color"]["reward"] for r in results)}
    (args.output / "calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
