"""Export frozen release image features; run with uv --project /root/inkling-mini.

The feature/mask pair is the trainer input contract; full prompt embeddings are
also exported for the currently validated vLLM prompt-embedding interface.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    revision = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
    source = "thinkingmachines/Inkling-2-Mini"
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        source, revision=revision, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    output = args.dataset / "features"
    output.mkdir(exist_ok=True)
    records = []
    for row in json.loads((args.dataset / "manifest.json").read_text())["records"]:
        path = Path(row["image"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed: {path}")
        with Image.open(path) as original:
            image = original.convert("RGB")
        original_size = list(image.size)
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            reasoning_effort="none",
        ).to(model.device)
        with torch.inference_mode():
            text = model.model.get_input_embeddings()(inputs["input_ids"])
            encoded = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
            features = torch.cat(encoded.pooler_output).to(text)
            mask = inputs["input_ids"] == model.config.image_token_id
            if mask.sum().item() != len(features) or not torch.isfinite(features).all():
                raise ValueError(f"Invalid image features: {row['id']}")
            prepared = text.masked_scatter(mask.unsqueeze(-1), features)
        target = output / f"{row['id']}.pt"
        torch.save(
            {
                "input_ids": inputs["input_ids"].cpu(),
                "image_features": features.cpu(),
                "image_token_mask": mask.cpu(),
                "prompt_embeds": prepared[0].cpu(),
                "image_grid_thw": inputs["image_grid_thw"].cpu(),
            },
            target,
        )
        records.append(
            {
                "id": row["id"],
                "split": row["split"],
                "features": str(target.resolve()),
                "source_image_sha256": row["image_sha256"],
                "original_size": original_size,
                "encoder_input_size": list(image.size),
                "resize": "aspect-preserving thumbnail max640 Lanczos",
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "image_tokens": len(features),
                "prompt_tokens": inputs["input_ids"].numel(),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "model": source,
                "revision": revision,
                "encoder_frozen": True,
                "contract": "text embed_norm first, then image features; do not renormalize features",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
