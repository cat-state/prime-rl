"""Pin NGA official CSVs; build bounded painting metadata and image manifests."""

import argparse
import csv
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from build_expanded import fetch, sha
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--images", type=int, default=100)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import shutil

    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free")
    (root / "images").mkdir(exist_ok=True)
    revision_path = root / "revision.json"
    if not revision_path.exists():
        info = json.loads(fetch("https://api.github.com/repos/NationalGalleryOfArt/opendata/commits/main"))
        revision_path.write_text(json.dumps({"revision": info["sha"]}))
    revision = json.loads(revision_path.read_text())["revision"]
    base = f"https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/{revision}/data/"
    metadata_path = root / "curated_metadata.json"
    if metadata_path.exists():
        candidates = json.loads(metadata_path.read_text())
    else:
        with urlopen(base + "objects.csv", timeout=60) as response:
            objects = {
                r["objectid"]: r
                for r in csv.DictReader(io.TextIOWrapper(response, encoding="utf-8-sig"))
                if r["classification"] == "Painting" and r["accessioned"] == "1"
            }
        print(f"Found {len(objects)} catalog paintings", flush=True)
        candidates, seen = [], set()
        with urlopen(base + "published_images.csv", timeout=60) as response:
            for image in csv.DictReader(io.TextIOWrapper(response, encoding="utf-8-sig")):
                oid = image["depictstmsobjectid"]
                if oid not in objects or oid in seen or image["openaccess"] != "1" or image["viewtype"] != "primary":
                    continue
                seen.add(oid)
                obj = objects[oid]
                key = obj.get("wikidataid") or f"nga:{oid}"
                candidates.append(
                    {
                        "id": f"nga-{oid}",
                        "artwork_key": key,
                        "title": obj["title"],
                        "artist": obj["attribution"],
                        "date": obj["displaydate"],
                        "medium": obj["medium"],
                        "source": f"https://www.nga.gov/artworks/{oid}",
                        "metadata_source": base,
                        "metadata_revision": revision,
                        "metadata_sha256": sha(
                            json.dumps({"object": obj, "published_image": image}, sort_keys=True).encode()
                        ),
                        "catalog_object": obj,
                        "published_image": image,
                        "download_url": image["iiifurl"] + "/full/!768,768/0/default.jpg",
                        "is_public_domain": True,
                        "rights_basis": "NGA published_images openaccess=1",
                        "rights_policy": "https://www.nga.gov/open-access-images.html",
                        "split": "heldout" if int(sha(key.encode())[:8], 16) % 5 == 0 else "train",
                        "image": None,
                        "image_sha256": None,
                    }
                )
        candidates.sort(key=lambda r: sha(r["artwork_key"].encode()))
        metadata_path.write_text(json.dumps(candidates))
    artists, keys, rows = Counter(), set(), []
    for row in candidates:
        if artists[row["artist"]] >= 30 or row["artwork_key"] in keys:
            continue
        rows.append(dict(row))
        artists[row["artist"]] += 1
        keys.add(row["artwork_key"])
        if len(rows) == args.count:
            break
    # More restrictive artist cap for the downloaded seed than the metadata bank.
    selected, image_artists = [], Counter()
    for row in rows:
        if image_artists[row["artist"]] < 8:
            selected.append(row)
            image_artists[row["artist"]] += 1
        if len(selected) == args.images:
            break

    def download(row):
        path = root / "images" / (row["id"] + ".jpg")
        if not path.exists():
            data = fetch(row["download_url"])
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            path.write_bytes(data)
        with Image.open(path) as image:
            image.load()
            row["dimensions"] = list(image.size)
        row.update(image=str(path), image_sha256=sha(path.read_bytes()))
        return row["id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, oid in enumerate(pool.map(download, selected), 1):
            print(f"Downloaded {index}/{len(selected)}: {oid}", flush=True)
    image_rows = [r for r in rows if r["image"]]
    if len({r["image_sha256"] for r in image_rows}) != len(image_rows):
        raise ValueError("Duplicate images detected; review before splitting")
    for split in ("train", "heldout"):
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in image_rows if r["split"] == split))
    (root / "manifest.json").write_text(
        json.dumps(
            {"revision": revision, "records": rows, "complete": len(rows) == args.count, "downloaded": len(image_rows)},
            indent=2,
        )
    )
    print(json.dumps({"records": len(rows), "images": len(image_rows), "artists": len(artists)}))


if __name__ == "__main__":
    main()
