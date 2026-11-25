"""Deterministic small color-identification task and strict final-answer reward."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def reward(raw_output, answer, finish_reason):
    if finish_reason != "stop" or "<|content_text|>" not in raw_output:
        return 0.0
    final = raw_output.rsplit("<|content_text|>", 1)[1].split("<|end_message|>", 1)[0]
    return float(final.strip().lower().rstrip(".") == answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    records = []
    palette = {"red": [(240, 20, 20), (210, 35, 40)], "blue": [(20, 30, 240), (35, 55, 205)],
               "green": [(20, 210, 30), (35, 180, 45)], "yellow": [(240, 230, 20), (215, 205, 35)]}
    for color, shades in palette.items():
        for split, rgb in zip(("train", "heldout"), shades, strict=True):
            key = f"{color}-{split}"
            path = args.directory / f"{key}.png"
            Image.new("RGB", (160, 160), rgb).save(path)
            records.append({"id": key, "split": split, "answer": color, "rgb": rgb,
                            "image": str(path.resolve()), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "prompt": "What color is this image? Choose red, blue, green, or yellow. Answer with exactly one color word."})
    (args.directory / "manifest.json").write_text(json.dumps({"task": "solid-color-v1", "records": records}, indent=2))


if __name__ == "__main__":
    main()
