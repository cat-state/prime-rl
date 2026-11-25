"""Compare complete, paired image evaluations without treating seeds as new images."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("updated", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    summaries = [json.loads((path / "summary.json").read_text()) for path in (args.baseline, args.updated)]
    indexes = [{(r["id"], r["sample"]): r for r in rows} for rows in summaries]
    if any(len(rows) != args.expected or len(index) != args.expected for rows, index in zip(summaries, indexes)):
        raise ValueError("Evaluation incomplete or duplicate sample IDs")
    if indexes[0].keys() != indexes[1].keys():
        raise ValueError("Evaluation sample sets differ")
    images = {}
    for key in sorted(indexes[0]):
        records = [json.loads((path / f"{key[0]}-{key[1]}.json").read_text()) for path in (args.baseline, args.updated)]
        for field in ("prompt_id", "seed", "temperature", "max_tokens"):
            if records[0]["request"][field] != records[1]["request"][field]:
                raise ValueError(f"Unpaired {field}: {key}")
        for field in ("source_image_sha256", "prompt_token_ids"):
            if records[0]["response"][field] != records[1]["response"][field]:
                raise ValueError(f"Different {field}: {key}")
        stats = images.setdefault(key[0], {"samples": 0, "correct": [0, 0], "truncated": [0, 0],
                                         "tokens": [0, 0], "improved": 0, "regressed": 0})
        stats["samples"] += 1
        scores = [r["reward"] for r in records]
        stats["improved"] += scores[1] > scores[0]
        stats["regressed"] += scores[1] < scores[0]
        for i, record in enumerate(records):
            stats["correct"][i] += record["reward"]
            stats["truncated"][i] += record["response"]["finish_reason"] == "length"
            stats["tokens"][i] += len(record["response"]["completion_ids"])
    result = {"policy_order": [str(args.baseline), str(args.updated)], "by_image": images,
              "limitation": "Seeds repeat the same images; they are not independent held-out image coverage."}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
