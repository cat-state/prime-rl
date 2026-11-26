"""Bounded, resumable Met painting curation; all assets stay in the output directory."""

import argparse
import hashlib
import io
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

API = "https://collectionapi.metmuseum.org/public/collection"
QUERIES = [
    "landscape",
    "portrait",
    "still life",
    "city",
    "sea",
    "flowers",
    "interior",
    "religious",
    "Chinese",
    "Japanese",
    "Indian",
    "Dutch",
    "Italian",
    "French",
    "American",
    "Spanish",
]
LEGACY_SPLITS = {
    436535: "train",
    436532: "train",
    436524: "train",
    436534: "train",
    436105: "heldout",
    437881: "heldout",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "PaintingRLResearch/0.2"}), timeout=35) as response:
                data = response.read(12_000_001)
                if len(data) > 12_000_000:
                    raise ValueError("Response exceeds 12 MB download bound")
                return data
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--artist-cap", type=int, default=8)
    p.add_argument("--search-limit", type=int, default=100)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; no downloads performed")
    for name in ("metadata", "images", "search"):
        (root / name).mkdir(exist_ok=True)
    errors = []
    pools = []
    for query in QUERIES:
        path = root / "search" / f"{query.replace(' ', '-')}.json"
        url = (
            API
            + "/v1.1/search?"
            + urlencode({"hasImages": "true", "medium": "Paintings", "q": query, "limit": args.search_limit})
        )
        try:
            if not path.exists():
                path.write_bytes(fetch(url))
            pools.append(json.loads(path.read_text()).get("objectIDs") or [])
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append({"url": url, "error": str(exc)})
    ids = list(LEGACY_SPLITS)
    for i in range(max(map(len, pools), default=0)):
        ids.extend(pool[i] for pool in pools if i < len(pool))
    ids = list(dict.fromkeys(ids))
    (root / "candidate_ids.json").write_text(json.dumps(ids))

    def get_meta(object_id):
        path = root / "metadata" / f"met-{object_id}.json"
        url = f"{API}/v1/objects/{object_id}"
        try:
            if not path.exists():
                path.write_bytes(fetch(url))
            return object_id, json.loads(path.read_text()), None
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            return object_id, None, {"url": url, "error": str(exc)}

    rows, artists, hashes = [], Counter(), set()
    # Ordered batches preserve query interleaving and deterministic selection.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(ids), 16):
            for object_id, meta, error in pool.map(get_meta, ids[start : start + 16]):
                if error:
                    errors.append(error)
                    continue
                if (
                    not meta.get("isPublicDomain")
                    or meta.get("objectName", "").lower() != "painting"
                    or not meta.get("primaryImageSmall")
                ):
                    continue
                artist = meta.get("artistDisplayName") or "Unknown"
                if artists[artist] >= args.artist_cap:
                    continue
                artwork_key = meta.get("objectWikidata_URL") or f"met:{object_id}"
                split = LEGACY_SPLITS.get(
                    object_id, "heldout" if int(sha(artwork_key.encode())[:8], 16) % 5 == 0 else "train"
                )
                row = {
                    "id": f"met-{object_id}",
                    "artwork_key": artwork_key,
                    "split": split,
                    "title": meta["title"],
                    "artist": artist,
                    "date": meta.get("objectDate"),
                    "medium": meta.get("medium"),
                    "culture": meta.get("culture"),
                    "source": meta["objectURL"],
                    "metadata_url": f"{API}/v1/objects/{object_id}",
                    "download_url": meta["primaryImageSmall"],
                    "is_public_domain": True,
                    "rights_policy": "https://www.metmuseum.org/hubs/open-access",
                    "metadata_sha256": sha((root / "metadata" / f"met-{object_id}.json").read_bytes()),
                    "image": None,
                    "image_sha256": None,
                }
                if len(rows) < args.images:
                    path = root / "images" / f"met-{object_id}.jpg"
                    try:
                        if not path.exists():
                            data = fetch(row["download_url"])
                            with Image.open(io.BytesIO(data)) as image:
                                image.verify()
                            path.write_bytes(data)
                        digest = sha(path.read_bytes())
                        if digest in hashes:
                            continue
                        with Image.open(path) as image:
                            image.load()
                            if min(image.size) < 128 or max(image.size) / min(image.size) > 4:
                                continue
                            row["dimensions"] = list(image.size)
                        row.update(image=str(path), image_sha256=digest)
                        hashes.add(digest)
                    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                        errors.append({"url": row["download_url"], "error": str(exc)})
                        continue
                rows.append(row)
                artists[artist] += 1
                print(f"{len(rows)}/{args.count} {split} {artist}: {row['title']}", flush=True)
                if len(rows) >= args.count:
                    break
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "selection": vars(args) | {"output": str(root)},
                        "records": rows,
                        "complete": len(rows) >= args.count,
                    },
                    indent=2,
                )
            )
            (root / "errors.json").write_text(json.dumps(errors, indent=2))
            for split in ("train", "heldout"):
                (root / f"{split}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows if r["split"] == split and r["image"])
                )
            if len(rows) >= args.count:
                break
    print(
        json.dumps(
            {
                "usable_images": sum(bool(r["image"]) for r in rows),
                "records": len(rows),
                "artists": len(artists),
                "errors": len(errors),
            }
        )
    )


if __name__ == "__main__":
    main()
