"""Build non-overlapping red-dot counting images with blue distractors."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    records = []
    for split, seed_offset in (("train", 100), ("heldout", 200)):
        for index, count in enumerate((7, 9, 11, 13)):
            seed = seed_offset + index
            rng = random.Random(seed)
            cells = [(x, y) for y in range(5) for x in range(6)]
            rng.shuffle(cells)
            image = Image.new("RGB", (360, 300), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            dots = []
            for j, (x, y) in enumerate(cells[:count + 7]):
                cx, cy = x * 60 + 30 + rng.randint(-8, 8), y * 60 + 30 + rng.randint(-8, 8)
                radius = rng.randint(11, 17)
                color = "red" if j < count else "blue"
                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                             fill=(225, 30, 35) if color == "red" else (30, 60, 225))
                dots.append({"center": [cx, cy], "radius": radius, "color": color})
            key = f"dots-{split}-{index}"
            path = args.directory / f"{key}.png"
            image.save(path)
            records.append({"id": key, "split": split, "answer": str(count), "seed": seed, "dots": dots,
                            "image": str(path.resolve()), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "prompt": "How many red circles are in this image? Ignore blue circles. Answer with exactly one integer and no other text."})
    (args.directory / "manifest.json").write_text(json.dumps({"task": "red-dot-counting-v1", "records": records}, indent=2))


if __name__ == "__main__":
    main()
