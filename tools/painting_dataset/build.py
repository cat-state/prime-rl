"""Fetch a small provenance-bearing public-domain painting dataset from the Met."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

# Split by artwork, never by crop. All derivatives inherit this assignment.
ARTWORKS = {436535: "train", 436532: "train", 436524: "train", 436534: "train", 436105: "heldout", 437881: "heldout"}
PROMPT = "Reconstruct the provided painting as a procedural WGSL scene. Return only WGSL defining fn scene_override(p: vec3f) -> vec2f (signed distance, material ID). Use the supplied renderer helpers and palette. Do not sample the target image or include external resources."


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "PaintingRLResearch/0.1"}), timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for object_id, split in ARTWORKS.items():
        metadata_path = args.output / f"met-{object_id}.json"
        api = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        if not metadata_path.exists():
            metadata_path.write_bytes(fetch(api))
        meta = json.loads(metadata_path.read_text())
        if not meta["isPublicDomain"] or meta["objectName"] != "Painting":
            raise ValueError(f"Not a verified public-domain painting: {object_id}")
        image_url = meta["primaryImageSmall"]
        if not image_url:
            raise ValueError(f"Missing image: {object_id}")
        image_path = args.output / f"met-{object_id}.jpg"
        if not image_path.exists():
            data = fetch(image_url)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            image_path.write_bytes(data)
        with Image.open(image_path) as image:
            dimensions = list(image.size)
        rows.append(
            {
                "id": f"met-{object_id}",
                "split": split,
                "title": meta["title"],
                "artist": meta["artistDisplayName"],
                "date": meta["objectDate"],
                "image": str(image_path.resolve()),
                "dimensions": dimensions,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "source": meta["objectURL"],
                "metadata_url": api,
                "download_url": image_url,
                "is_public_domain": True,
                "rights_policy": "https://www.metmuseum.org/about-the-met/policies-and-documents/open-access",
                "prompt": PROMPT,
                "target_code": None,
            }
        )
        print(f"{split}: {meta['title']}", flush=True)
    for split in ("train", "heldout"):
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows if row["split"] == split)
        )
    (args.output / "manifest.json").write_text(
        json.dumps({"version": 1, "source": "Met Open Access API", "split_unit": "artwork", "records": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
