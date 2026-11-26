"""CPU diagnostic controls for a frozen painting reward; never promotes it to RL."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from tools.painting_dataset.painting_reward import score


def variants(target, wrong, seed):
    pixels = np.asarray(target)
    rgb = pixels.astype(np.float64) / 255
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    design = np.stack([np.ones_like(x), x / max(w - 1, 1), y / max(h - 1, 1)], axis=-1).reshape(-1, 3)
    fit = design @ np.linalg.lstsq(design, rgb.reshape(-1, 3), rcond=None)[0]
    gradient = Image.fromarray(np.rint(np.clip(fit.reshape(h, w, 3), 0, 1) * 255).astype(np.uint8))
    rng = np.random.default_rng(seed)
    block = (
        np.asarray(target.resize((128, 128))).reshape(8, 16, 8, 16, 3).transpose(0, 2, 1, 3, 4).reshape(64, 16, 16, 3)
    )
    block = block[rng.permutation(64)].reshape(8, 8, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(128, 128, 3)
    output = {
        "resized_reference": target,
        "blur2": target.filter(ImageFilter.GaussianBlur(2)),
        "blur8": target.filter(ImageFilter.GaussianBlur(8)),
        "blur24": target.filter(ImageFilter.GaussianBlur(24)),
        "mean_color": Image.new("RGB", target.size, tuple(np.rint(pixels.mean(axis=(0, 1))).astype(int))),
        "black": Image.new("RGB", target.size),
        "wrong_artwork": wrong.resize(target.size),
        "shuffled_pixels": Image.fromarray(rng.permutation(pixels.reshape(-1, 3)).reshape(pixels.shape)),
        "fitted_gradient": gradient,
        "shuffled_blocks": Image.fromarray(block).resize(target.size),
    }
    for grid in (2, 4, 8, 16):
        output[f"mosaic{grid}"] = target.resize((grid, grid), Image.Resampling.BOX).resize(
            target.size, Image.Resampling.NEAREST
        )
    return output


def load(path):
    with Image.open(path) as image:
        result = image.convert("RGB")
    result.thumbnail((256, 256))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--episodes", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in json.loads(args.manifest.read_text())["records"] if r.get("image")]
    records, errors = [], []
    comparisons = [
        ("resized_reference", "blur2"),
        ("blur2", "blur8"),
        ("blur8", "blur24"),
        ("blur8", "mean_color"),
        ("blur8", "fitted_gradient"),
        ("mosaic16", "mean_color"),
        ("mosaic16", "fitted_gradient"),
        ("mosaic8", "mean_color"),
        ("mosaic8", "fitted_gradient"),
        ("mosaic16", "shuffled_blocks"),
    ]
    with tempfile.TemporaryDirectory(prefix="controls-", dir=args.output) as temp:
        folder = Path(temp)
        for i, row in enumerate(rows):
            target = load(row["image"])
            controls = variants(
                target,
                load(rows[(i + 1) % len(rows)]["image"]),
                int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16),
            )
            metrics = {"identity": score(row["image"], row["image"])}
            for name, image in controls.items():
                path = folder / f"{name}.png"
                image.save(path)
                try:
                    metrics[name] = score(row["image"], path)
                except (ValueError, OSError) as error:
                    errors.append({"id": row["id"], "control": name, "error": str(error)})
            failures = [
                {"expected_higher": a, "expected_lower": b, "margin": metrics[a]["reward"] - metrics[b]["reward"]}
                for a, b in comparisons
                if a in metrics and b in metrics and metrics[a]["reward"] <= metrics[b]["reward"]
            ]
            rankings = {
                component: sorted(metrics, key=lambda name: metrics[name][component], reverse=component == "reward")
                for component in ("reward", "color_error", "edge_error")
            }
            records.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "image": row["image"],
                    "metrics": metrics,
                    "rankings": rankings,
                    "ordering_failures": failures,
                }
            )
            print(f"{i + 1}/{len(rows)} {row['id']}: {len(failures)} ordering flags", flush=True)
    aggregate = {
        name: {
            component: {
                "mean": float(np.mean([r["metrics"][name][component] for r in records if name in r["metrics"]])),
                "min": float(min(r["metrics"][name][component] for r in records if name in r["metrics"])),
                "max": float(max(r["metrics"][name][component] for r in records if name in r["metrics"])),
            }
            for component in ("reward", "color_error", "edge_error")
        }
        for name in records[0]["metrics"]
    }
    checks = {
        f"{a}>{b}": sum(
            r["metrics"][a]["reward"] > r["metrics"][b]["reward"]
            for r in records
            if a in r["metrics"] and b in r["metrics"]
        )
        for a, b in comparisons
    }

    def wash_margin(record):
        m = record["metrics"]
        return max(m["mean_color"]["reward"], m["fitted_gradient"]["reward"]) - m["mosaic16"]["reward"]

    worst = sorted(records, key=wash_margin, reverse=True)[:6]
    canvas = Image.new("RGB", (5 * 180, len(worst) * 220), "white")
    draw = ImageDraw.Draw(canvas)
    byid = {r["id"]: i for i, r in enumerate(rows)}
    for index, record in enumerate(worst):
        i = byid[record["id"]]
        target = load(record["image"])
        controls = variants(
            target,
            load(rows[(i + 1) % len(rows)]["image"]),
            int(hashlib.sha256(record["id"].encode()).hexdigest()[:8], 16),
        )
        for col, name in enumerate(("resized_reference", "mosaic16", "fitted_gradient", "mean_color", "blur8")):
            thumb = ImageOps.contain(controls[name], (176, 176))
            canvas.paste(thumb, (col * 180 + (180 - thumb.width) // 2, index * 220))
            draw.text(
                (col * 180 + 2, index * 220 + 178), f"{name}: {record['metrics'][name]['reward']:.3f}", fill="black"
            )
            draw.text((col * 180 + 2, index * 220 + 196), record["id"], fill="black")
    canvas.save(args.output / "worst-wash-comparisons.jpg", quality=90)
    actual = []
    if args.episodes:
        for episode in sorted(args.episodes.glob("ep-*")):
            target = episode / "target.png"
            for render_file in sorted(episode.glob("turn-*/render.json")):
                result = json.loads(render_file.read_text())
                if result.get("ok") and result.get("image"):
                    actual.append(
                        {
                            "episode": str(episode),
                            "turn": render_file.parent.name,
                            "image": result["image"],
                            "metrics": score(target, result["image"]),
                        }
                    )
    report = {
        "status": "DIAGNOSTIC ONLY; no RL reward promotion",
        "metric_source_sha256": hashlib.sha256(Path(__file__).with_name("painting_reward.py").read_bytes()).hexdigest(),
        "reference_preprocessing": "Score full-resolution museum reference; candidates max256 per axis. Identity uses original bytes; resized_reference quantifies sampling loss.",
        "records": records,
        "aggregate": aggregate,
        "ordering_pass_counts": checks,
        "errors": errors,
        "actual_candidates": actual,
        "worst_wash_ids": [r["id"] for r in worst],
        "warning": "Mosaic and blur orderings are screening proxies, not human aesthetic labels. Contact sheet requires visual review.",
    }
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "records": len(records),
                "errors": len(errors),
                "ordering_pass_counts": checks,
                "actual_candidates": actual,
            }
        )
    )


if __name__ == "__main__":
    main()
