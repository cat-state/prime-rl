"""Summarize completed rollout validity and context pressure without loading route tensors."""
import argparse
import json
from pathlib import Path


def summarize(root):
    timing = json.loads((root / "timing.json").read_text())
    if "episodes" not in timing:
        raise ValueError("Run is not complete")
    records = []
    for episode in timing["episodes"]:
        ep = Path(episode)
        state = json.loads((ep / "episode.json").read_text())
        if not state.get("complete"):
            raise ValueError("Incomplete episode")
        for turn in state["turns"]:
            directory = ep / f"turn-{turn['turn']}"
            response = json.loads((directory / "response.json").read_text())
            encoded = json.loads((directory / "encoded.json").read_text())
            records.append({"episode": str(ep), "seed": state["seed"], "turn": turn["turn"],
                            "target_sha256": state["target_sha256"], "valid": turn["valid"],
                            "error": turn.get("error"), "prompt_tokens": len(response["prompt_token_ids"]),
                            "completion_tokens": len(response["completion_ids"]),
                            "finish_reason": response["finish_reason"], "images": len(encoded["image_spans"]),
                            "history_policy": state["config"].get("history_policy", "full-v1")})
    turns = []
    for index in sorted({r["turn"] for r in records}):
        rows = [r for r in records if r["turn"] == index]
        turns.append({"turn": index, "n": len(rows), "valid": sum(r["valid"] for r in rows),
                      "truncations": sum(r["finish_reason"] == "length" for r in rows),
                      "prompt_min": min(r["prompt_tokens"] for r in rows),
                      "prompt_max": max(r["prompt_tokens"] for r in rows),
                      "prompt_mean": sum(r["prompt_tokens"] for r in rows)/len(rows),
                      "max_images": max(r["images"] for r in rows)})
    tokens = sum(r["completion_tokens"] for r in records)
    return {"root": str(root.resolve()), "episodes": len(timing["episodes"]), "turns": turns,
            "tokens": tokens, "wall_seconds": timing["wall_seconds"],
            "tokens_per_second": tokens/timing["wall_seconds"], "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
