"""Persistent frozen release encoder, run in the preserved inkling-mini environment."""
import argparse
import hashlib
import json
import socket
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

REVISION = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
SOURCE = "thinkingmachines/Inkling-2-Mini"


def encode(model, processor, messages, directory):
    transformed, images = [], []
    for message in messages:
        if message["role"] == "assistant":
            text = message["content"]
            if "<|content_text|>" in text:
                text = text.split("<|content_text|>", 1)[1].split("<|end_message|>", 1)[0]
            transformed.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            continue
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                path = Path(part["path"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != part["sha256"]:
                    raise ValueError("Input image changed")
                with Image.open(path) as original:
                    image = original.convert("RGB")
                image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                content.append({"type": "image", "image": image})
                images.append((path, digest))
        transformed.append({"role": message["role"], "content": content})
    inputs = processor.apply_chat_template(transformed, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", reasoning_effort="none").to(model.device)
    with torch.inference_mode():
        text = model.model.get_input_embeddings()(inputs["input_ids"])
        encoded = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
        features = torch.cat(encoded.pooler_output).to(text)
        mask = inputs["input_ids"] == model.config.image_token_id
        assert mask.sum().item() == len(features) and torch.isfinite(features).all()
        prepared = text.masked_scatter(mask.unsqueeze(-1), features)
    path = directory / "prompt.pt"
    torch.save({"input_ids": inputs["input_ids"].cpu(), "image_features": features.cpu(),
                "image_token_mask": mask.cpu(), "prompt_embeds": prepared[0].cpu()}, path)
    positions = mask[0].nonzero().flatten().cpu().tolist()
    spans = []
    for position in positions:
        if spans and position == spans[-1][0] + spans[-1][1]:
            spans[-1][1] += 1
        else:
            spans.append([position, 1])
    if len(spans) != len(images):
        raise ValueError("Image token spans do not match image order")
    records, metadata, offset = [], [], 0
    for i, ((start, count), (image, digest)) in enumerate(zip(spans, images, strict=True)):
        feature_path = directory / f"image-{i}.pt"
        torch.save({"image_features": features[offset:offset + count].cpu()}, feature_path)
        offset += count
        metadata.append({"start": start, "count": count, "image": str(image), "image_sha256": digest})
        # Repeated identical images must map to one canonical cache path.
        if not any(r["source_image_sha256"] == digest for r in records):
            records.append({"source_image_sha256": digest, "features": str(feature_path)})
    manifest = directory / "features-manifest.json"
    manifest.write_text(json.dumps({"encoder_frozen": True, "model": SOURCE, "revision": REVISION, "records": records}))
    return {"features": str(path), "feature_manifest": str(manifest), "input_ids": inputs["input_ids"][0].cpu().tolist(),
            "mm_token_type_ids": mask[0].long().cpu().tolist(), "image_spans": metadata,
            "encoder_revision": REVISION}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    processor = AutoProcessor.from_pretrained(SOURCE, revision=REVISION)
    model = AutoModelForMultimodalLM.from_pretrained(SOURCE, revision=REVISION, dtype=torch.bfloat16, device_map="cuda:0").eval()
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(args.socket))
        args.socket.chmod(0o600)
        server.listen(1)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    request = json.loads(connection.makefile("rb").readline(8_000_000))
                    directory = Path(request["directory"]).resolve(strict=True)
                    if not directory.is_relative_to(root):
                        raise ValueError("Encoder destination outside workspace")
                    for message in request["messages"]:
                        if isinstance(message["content"], list):
                            for part in message["content"]:
                                if part["type"] == "image" and not Path(part["path"]).resolve(strict=True).is_relative_to(root):
                                    raise ValueError("Encoder image outside workspace")
                    result = {"ok": True, "encoded": encode(model, processor, request["messages"], directory)}
                except Exception as error:
                    result = {"ok": False, "error": repr(error)}
                connection.sendall((json.dumps(result) + "\n").encode())


if __name__ == "__main__":
    main()
