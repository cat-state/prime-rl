"""Probe image input in the preserved release-reference environment, not PrimeRL.

Run with uv run --project /root/inkling-mini --no-sync python THIS_FILE OUTPUT.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = "thinkingmachines/Inkling-2-Mini"
revision = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
processor = AutoProcessor.from_pretrained(source, revision=revision)
model = AutoModelForMultimodalLM.from_pretrained(source, revision=revision, dtype=torch.bfloat16, device_map="cuda:0")
results = []
for color in ("red", "blue"):
    picture = Image.new("RGB", (160, 160), color)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {"type": "text", "text": "What is the solid color in this image? Answer with one color word."},
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
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        embeds = model.model.get_input_embeddings()(inputs["input_ids"])
        image_features = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
        features = torch.cat(image_features.pooler_output).to(embeds)
        mask = model.model.get_placeholder_mask(inputs["input_ids"], embeds, features, model.config.image_token_id)
        embeds = embeds.masked_scatter(mask, features)
        torch.save(
            {"prompt_embeds": embeds[0].cpu(), "input_ids": inputs["input_ids"][0].cpu()},
            args.output.with_name(f"image-reference-{color}-embeds.pt"),
        )
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    torch.cuda.synchronize()
    generated = outputs[0, inputs["input_ids"].shape[-1] :]
    results.append(
        {
            "input_color": color,
            "input_shapes": {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            "output": processor.decode(generated, skip_special_tokens=False),
            "generated_tokens": len(generated),
            "seconds": time.monotonic() - start,
        }
    )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results[-1]), flush=True)
