"""Collect bounded PD12M painting candidates; metadata only, not training-ready."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from build_expanded import fetch, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--rows", type=int, default=1000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = json.loads(fetch("https://huggingface.co/api/datasets/Spawning/PD12M"))
    revision = info["sha"]
    candidates, seen = [], set()
    pages = []
    for offset in range(0, args.rows, 100):
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {
                "dataset": "Spawning/PD12M",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": min(100, args.rows - offset),
            }
        )
        data = fetch(url)
        page = json.loads(data)
        pages.append({"url": url, "sha256": sha(data)})
        for item in page["rows"]:
            row = item["row"]
            caption = row["caption"].lower()
            if "painting" not in caption or row["hash"] in seen:
                continue
            if row["license"] not in (
                "https://creativecommons.org/publicdomain/zero/1.0/",
                "https://creativecommons.org/publicdomain/mark/1.0/",
            ):
                continue
            seen.add(row["hash"])
            candidates.append(
                row
                | {
                    "dataset": "Spawning/PD12M",
                    "observed_revision": revision,
                    "row_index": item["row_idx"],
                    "status": "candidate_metadata_only",
                    "split": "unassigned",
                    "artwork_identity": None,
                    "image_sha256": None,
                }
            )
    after = json.loads(fetch("https://huggingface.co/api/datasets/Spawning/PD12M"))["sha"]
    if after != revision:
        raise RuntimeError("Dataset revision changed during metadata collection")
    (args.output / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "dataset": "Spawning/PD12M",
                "observed_revision": revision,
                "license": "CDLA-Permissive-2.0",
                "rows_scanned": args.rows,
                "candidates": len(candidates),
                "pages": pages,
                "warning": "Viewer rows are not revision-addressed. Revision checked before/after and page hashes recorded. Caption filter does not establish artwork identity, painter, public museum catalog link, or artwork-disjoint split. No image files downloaded; supplied MD5 is unverified.",
            },
            indent=2,
        )
    )
    print(f"Staged {len(candidates)} metadata-only painting candidates from {args.rows} rows")


if __name__ == "__main__":
    main()
