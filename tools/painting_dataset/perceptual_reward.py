"""Pinned DINOv2 aligned spatial features; diagnostic until candidate calibration passes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageOps
from transformers import AutoModel

from tools.painting_dataset.review_reward import load, variants

MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
LAYERS = (3, 6, 9, 12)


class SpatialPerceptual:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval().to(self.device)
        self.model.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    def prepare(self, image):
        image = image.convert("RGB")
        image.thumbnail((224, 224), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (224, 224), (124, 116, 104))
        x, y = (224 - image.width) // 2, (224 - image.height) // 2
        canvas.paste(image, (x, y))
        mask = np.zeros((224, 224), dtype=np.float32)
        mask[y : y + image.height, x : x + image.width] = 1
        pixels = np.asarray(canvas, dtype=np.float32) / 255
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1), torch.from_numpy(mask)

    @torch.inference_mode()
    def features(self, images):
        prepared = [self.prepare(image) for image in images]
        pixels = torch.stack([p[0] for p in prepared]).to(self.device)
        masks = torch.stack([p[1] for p in prepared]).to(self.device)
        weights = F.avg_pool2d(masks[:, None], 14, 14).flatten(1)
        result = self.model(pixel_values=(pixels - self.mean) / self.std, output_hidden_states=True)
        states = [result.hidden_states[index] if index != 12 else result.last_hidden_state for index in LAYERS]
        patches = torch.stack([F.normalize(state[:, 1:].float(), dim=-1) for state in states], dim=1)
        cls = F.normalize(result.last_hidden_state[:, 0].float(), dim=-1)
        return patches.cpu(), cls.cpu(), weights.cpu()

    def compare(self, images):
        patches, cls, weights = self.features(images)
        weights = torch.minimum(weights[0:1], weights)
        cosine = (patches[0:1] * patches).sum(-1)
        spatial = (cosine * weights[:, None]).sum(-1) / weights.sum(-1)[:, None].clamp_min(1e-8)
        global_cosine = (cls[0:1] * cls).sum(-1)
        return [
            {
                "spatial_cosine": float(spatial[i].mean()),
                "layer_cosines": spatial[i].tolist(),
                "last_patch_cosine": float(spatial[i, -1]),
                "cls_cosine": float(global_cosine[i]),
            }
            for i in range(len(images))
        ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--episodes", type=Path, action="append", default=[])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metric = SpatialPerceptual(args.device)
    rows = [r for r in json.loads(args.manifest.read_text())["records"] if r.get("image")][: args.limit]
    records = []
    for i, row in enumerate(rows):
        target = load(row["image"])
        controls = variants(
            target,
            load(rows[(i + 1) % len(rows)]["image"]),
            int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16),
        )
        controls.update(
            horizontal_flip=ImageOps.mirror(target),
            rotate180=target.transpose(Image.Transpose.ROTATE_180),
            shift1=ImageChops.offset(target, 1, 0),
            shift32=ImageChops.offset(target, 32, 0),
        )
        names = ["identity"] + list(controls)
        scores = metric.compare([target] + list(controls.values()))
        records.append({"id": row["id"], "metrics": dict(zip(names, scores, strict=True))})
        print(f"{i + 1}/{len(rows)} {row['id']}", flush=True)
    actual = []
    for directory in args.episodes:
        for episode in sorted(directory.glob("ep-*")):
            target = load(episode / "target.png")
            controls = variants(target, target, 401)
            images, names = [target], ["identity"]
            for name in [
                "mean_color",
                "fitted_gradient",
                "mosaic8",
                "mosaic16",
                "blur2",
                "blur8",
                "shuffled_pixels",
                "shuffled_blocks",
            ]:
                images.append(controls[name])
                names.append(name)
            for path in sorted(episode.glob("turn-*/render.json")):
                result = json.loads(path.read_text())
                if result.get("ok") and result.get("image"):
                    images.append(load(result["image"]))
                    names.append(path.parent.name)
            actual.append({"episode": str(episode), "metrics": dict(zip(names, metric.compare(images), strict=True))})
    checks = [
        ("identity", "blur2"),
        ("blur2", "blur8"),
        ("mosaic16", "mean_color"),
        ("mosaic16", "fitted_gradient"),
        ("mosaic8", "fitted_gradient"),
        ("mosaic16", "shuffled_blocks"),
        ("shift1", "shift32"),
        ("blur2", "horizontal_flip"),
        ("blur2", "wrong_artwork"),
    ]
    counts = {
        f"{a}>{b}": sum(r["metrics"][a]["spatial_cosine"] > r["metrics"][b]["spatial_cosine"] for r in records)
        for a, b in checks
    }
    report = {
        "status": "DIAGNOSTIC ONLY; no optimizer/reward promotion",
        "model": MODEL,
        "revision": REVISION,
        "layers": LAYERS,
        "preprocessing": "RGB preserve aspect to at most224x224, bicubic; mean-color letterbox; ImageNet normalization; aligned 14px patches weighted by valid image area. No CLS in spatial score.",
        "records": records,
        "ordering_pass_counts": counts,
        "actual_candidates": actual,
    }
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"ordering_pass_counts": counts, "actual_candidates": actual}))


if __name__ == "__main__":
    main()
