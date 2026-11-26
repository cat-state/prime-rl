"""Train-only fixed cached-prompt comparison of response caps; strict action extraction."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from tools.painting_dataset.agentic_episode import extract_action, validate_turn
from tools.painting_dataset.agentic_providers import SharedInferenceGenerator
from tools.painting_dataset.rgb_renderer import factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    timing = json.loads((args.pilot / "timing.json").read_text())
    jobs = json.loads((args.pilot / "jobs.json").read_text())
    candidates = []
    failures = []
    for job, episode in zip(jobs, timing["episodes"], strict=True):
        ep = Path(episode)
        state = json.loads((ep / "episode.json").read_text())
        for turn in state["turns"]:
            directory = ep / f"turn-{turn['turn']}"
            response = json.loads((directory / "response.json").read_text())
            row = {"directory": str(directory), "artwork_id": job["artwork_id"], "seed": response["seed"],
                   "target": str(ep / "target.png"), "finish_reason": response["finish_reason"], "valid": turn["valid"]}
            candidates.append(row)
            if not turn["valid"] and turn["error"] == "Expected exactly one final WGSL code block":
                raw = response["raw_output"]
                failures.append({**row, "wgsl_blocks": len(re.findall(r"```(?:wgsl)?\s*\n", raw)),
                                 "paint_definitions": len(re.findall(r"\bfn\s+paint\s*\(", raw)),
                                 "raw_output": raw, "note": "Read-only diagnostic; extraction remains strict, no repair."})
    truncated = [r for r in candidates if r["finish_reason"] == "length"]
    # Two flower failures plus two other targets where available; fixed by source order, not reward.
    flower = [r for r in truncated if r["artwork_id"] == "nga-126146"]
    selected = flower[:2]
    for row in truncated:
        if row not in selected and row["artwork_id"] != "nga-126146" and len(selected) < 4:
            selected.append(row)
    for row in truncated:
        if row not in selected and len(selected) < 4:
            selected.append(row)
    controls = []
    for row in candidates:
        if row["finish_reason"] == "stop" and row["valid"] and row["artwork_id"] not in {x["artwork_id"] for x in controls}:
            controls.append(row)
        if len(controls) == 4:
            break
    if len(selected) != 4 or len(controls) != 4:
        raise ValueError("Need four truncated and four naturally completed train prompts")
    selected += controls
    args.output.mkdir(exist_ok=False)
    (args.output / "raw-format-audit.json").write_text(json.dumps(failures, indent=2))
    encoded = [json.loads((Path(r["directory"]) / "encoded.json").read_text()) for r in selected]
    max_prompt = max(len(e["input_ids"]) for e in encoded)
    context = ((max_prompt + 16384 + 255)//256)*256
    for row in selected:
        f = Path(row["directory"]) / "response.json"
        row["original_response_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    provenance = {"rows": selected, "selection": "first two flower truncations, first two other train truncations; four first valid stops from distinct train artworks",
                  "max_prompt_tokens": max_prompt, "max_model_len": context, "kv_cache_gib": 56,
                  "batch_size": 8, "caps": [8192, 16384], "policy": "base", "temperature": .7,
                  "comparison": "BOTH arms newly generated in same resident engine/batch assignment; original responses selection only",
                  "storage_planner": "Shared TP2 weights ~25GiB/GPU +56GiB KV +~3GiB graph/workspace, below96GiB; SWA1024 limits local-cache growth. Startup planner must validate one full request; preemption warnings retained."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    generator = SharedInferenceGenerator(max_model_len=context, kv_cache_gib=56, tensor_parallel_size=2, max_num_seqs=8)
    summaries = []
    for cap in (8192, 16384):
        directory = args.output / f"cap-{cap}"
        directory.mkdir()
        requests = [(e, {"policy": "base", "temperature": .7, "seed": row["seed"], "max_tokens": cap})
                    for row, e in zip(selected, encoded, strict=True)]
        responses = generator.generate_batch(requests)
        records = []
        for index, (row, e, response) in enumerate(zip(selected, encoded, responses, strict=True)):
            workspace = directory / str(index)
            workspace.mkdir()
            (workspace / "response.json").write_text(json.dumps(response))
            validate_turn(response, e)
            ids = response["completion_ids"]
            marker = generator.tokenizer.convert_tokens_to_ids("<|content_text|>")
            split = ids.index(marker) + 1 if marker in ids else len(ids)
            record = {"index": index, "tokens": len(ids), "reasoning_and_prefix_tokens": split,
                      "final_tokens": len(ids)-split, "finish_reason": response["finish_reason"], "valid": False}
            try:
                source = extract_action(response, "paint")
                (workspace / "action.wgsl").write_text(source)
                with Image.open(row["target"]) as target:
                    scale = 256/max(target.size)
                    width, height = [max(1, round(x*scale)) for x in target.size]
                with factory(workspace) as renderer:
                    render = renderer.render(source, workspace, width, height)
                (workspace / "render.json").write_text(json.dumps(render, indent=2))
                record.update(valid=bool(render["ok"]), error=render.get("error"))
            except (ValueError, OSError, TimeoutError) as error:
                record["error"] = str(error)
            records.append(record)
        seconds = responses[0]["generation_seconds"]
        summary = {"cap": cap, "records": records, "generation_seconds": seconds,
                   "tokens": sum(r["tokens"] for r in records), "valid": sum(r["valid"] for r in records),
                   "natural_stops": sum(r["finish_reason"] == "stop" for r in records)}
        summary["tokens_per_second"] = summary["tokens"]/seconds
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
