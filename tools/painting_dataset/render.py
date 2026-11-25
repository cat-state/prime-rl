"""Render completed final-answer shaders; preserve raw responses and failures."""

import argparse
import json
import re
import socket
from pathlib import Path

import numpy as np
from PIL import Image


def extract(candidate):
    if candidate["finish_reason"] != "stop":
        raise ValueError("Incomplete generation")
    text = candidate["raw_output"].split("<|content_text|>", 1)[1].split("<|end_message|>", 1)[0]
    match = re.fullmatch(r"\s*```(?:wgsl)?\s*\n(.*?)\n```\s*", text, re.S)
    if not match:
        raise ValueError("Expected exactly one final WGSL block")
    source = match[1]
    if len(source) > 40000 or "fn scene_override" not in source or "@" in source:
        raise ValueError("Shader violates scene interface or source budget")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--socket", default="/tmp/painting-render.sock")
    args = parser.parse_args()
    rows = {r["id"]: r for r in json.loads((args.dataset / "manifest.json").read_text())["records"]}
    destination = args.candidates / "renders"
    destination.mkdir(exist_ok=True)
    for path in sorted(args.candidates.glob("*.json")):
        candidate = json.loads(path.read_text())
        record = {"id": candidate["id"], "valid": False}
        try:
            source = extract(candidate)
            with Image.open(rows[candidate["id"]]["image"]) as target:
                width, height = target.size
            scale = 256 / max(width, height)
            request = dict(scene=9, seed=42, width=round(width * scale), height=round(height * scale),
                           samples=1, eye=[0, 0, 4], look=[0, 0, 0], name=candidate["id"],
                           out=str(destination.resolve()), shader_src=source)
            (destination / f"{path.stem}.wgsl").write_text(source)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(100)
                client.connect(args.socket)
                client.sendall((json.dumps(request) + "\n").encode())
                response = json.loads(client.makefile("rb").readline())
            record["request"] = {k: v for k, v in request.items() if k != "shader_src"}
            record["response"] = response
            if response.get("ok"):
                arrays = [np.fromfile(f, dtype=np.float32) for f in response["files"] if f.endswith(".f32")]
                expected = request["width"] * request["height"]
                record["valid"] = bool(arrays) and all(a.size == expected and np.isfinite(a).all() for a in arrays)
                image_path = destination / f"{path.stem}-seed-42.png"
                pixels = np.asarray(Image.open(image_path).convert("RGB"))
                record["pixel_std"] = float(pixels.std())
                record["quality"] = "unscored; valid rendering does not establish painting similarity"
        except (ValueError, IndexError, OSError) as error:
            record["error"] = str(error)
        (destination / f"{path.stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
