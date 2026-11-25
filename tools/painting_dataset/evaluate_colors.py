"""Collect scored single-turn color rollouts, preserving full training evidence."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from tools.painting_dataset.colors import reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--policy", default="base")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.dataset / "manifest.json").read_text())["records"]
    required = {row["id"] for row in rows}
    for _ in range(60):
        try:
            health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2))
            if required.issubset(health["prompts"]):
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Requested task cache did not become ready")
    if args.adapter:
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/adapters",
                                         data=json.dumps({"policy": args.policy, "path": str(args.adapter.resolve())}).encode(),
                                         headers={"Content-Type": "application/json"})
        registration = json.load(urllib.request.urlopen(request, timeout=30))
        print(registration, flush=True)
    output = args.dataset / f"rollouts-{args.policy}"
    output.mkdir(exist_ok=False)
    summary = []
    for row in rows:
        for sample in range(4 if row["split"] == "train" else 1):
            payload = {"prompt_id": row["id"], "policy": args.policy, "temperature": 0.7,
                       "max_tokens": 512, "seed": 101 + sample}
            request = urllib.request.Request(f"http://127.0.0.1:{args.port}/generate",
                                             data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            response = json.load(urllib.request.urlopen(request, timeout=180))
            score = reward(response["raw_output"], row["answer"], response["finish_reason"])
            record = {"request": payload, "response": response, "reward": score,
                      "answer": row["answer"], "split": row["split"]}
            path = output / f"{row['id']}-{sample}.json"
            path.write_text(json.dumps(record))
            summary.append({"id": row["id"], "sample": sample, "reward": score, "split": row["split"],
                            "tokens": len(response["completion_ids"]), "finish_reason": response["finish_reason"]})
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(summary[-1], flush=True)


if __name__ == "__main__":
    main()
