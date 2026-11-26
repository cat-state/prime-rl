"""Render the built-in WGSL environment catalog with its default cameras."""

import argparse
import hashlib
import json
import re
import socket
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--socket", default="/tmp/painting-render.sock")
    args = parser.parse_args()
    names = re.findall(r'"([^"]+)"', args.source.read_text().split("const NAMES:", 1)[1].split("];", 1)[0])
    args.output.mkdir(parents=True, exist_ok=False)
    records, failures = [], []
    prompt = ("Describe this rendered image in 3–5 concise sentences. Identify the visible geometry, "
              "spatial layout, colors, materials, lighting, and viewpoint. Distinguish observations from guesses. "
              "Do not invent unseen objects or infer a title. Give your description directly.")
    for scene, name in enumerate(names, 1):
        key = f"environment-{scene:03d}"
        request = dict(scene=scene, seed=42, width=480, height=360, samples=2,
                       name=key, out=str(args.output.resolve()))
        try:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(150)
                client.connect(args.socket)
                client.sendall((json.dumps(request) + "\n").encode())
                response = json.loads(client.makefile("rb").readline())
            (args.output / f"{key}-render.json").write_text(json.dumps({"request": request, "response": response}, indent=2))
            if not response.get("ok"):
                raise ValueError(str(response))
            path = args.output / f"{key}-seed-42.png"
            pixels = np.asarray(Image.open(path).convert("RGB"))
            records.append(dict(id=key, scene=scene, scene_name=name, split="inspection", image=str(path.resolve()),
                                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                pixel_std=float(pixels.std()), prompt=prompt))
        except (OSError, ValueError) as error:
            failures.append(dict(scene=scene, name=name, error=str(error)))
        (args.output / "manifest.json").write_text(json.dumps(dict(records=records, failures=failures,
                                                                  catalog_size=len(names)), indent=2))
        print(dict(scene=scene, name=name, rendered=len(records), failures=len(failures)), flush=True)


if __name__ == "__main__":
    main()
