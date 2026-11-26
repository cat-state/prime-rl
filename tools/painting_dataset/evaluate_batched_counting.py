"""Paired heldout counting comparison using one temporary graph-enabled engine."""
import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tools.painting_dataset.colors import reward
from tools.painting_dataset.serve import adapter_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    rows = [r for r in json.loads((args.dataset / "manifest.json").read_text())["records"] if r["split"] == "heldout"]
    features = {r["id"]: r for r in json.loads((args.dataset / "features/manifest.json").read_text())["records"]}
    cache = {r["id"]: torch.load(features[r["id"]]["features"], map_location="cpu", weights_only=True) for r in rows}
    assert len(rows) == 4
    assignments = [(row, seed) for row in rows for seed in range(301, 309)]
    llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=2,
              enable_prompt_embeds=True, enforce_eager=False, max_num_seqs=32,
              max_model_len=2048, max_num_batched_tokens=4096,
              kv_cache_memory_bytes=24 * 1024**3, enable_prefix_caching=True,
              enable_flashinfer_autotune=False, tokenizer_mode="hf", enable_lora=True,
              max_lora_rank=8, lora_target_modules=["wo_ud"],
              aux_output_config={"enable_return_routed_experts": True})
    tokenizer = llm.get_tokenizer()
    stop = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    params = [SamplingParams(temperature=.7, seed=seed, max_tokens=1536, logprobs=0, stop_token_ids=[stop])
              for _, seed in assignments]
    prompts = [{"prompt_embeds": cache[row["id"]]["prompt_embeds"]} for row, _ in assignments]
    summary = {}
    for index, policy in enumerate(("base", "eager", "compiled")):
        adapter = args.dataset / ("adapter-group-step1" + ("-compiled" if policy == "compiled" else ""))
        request = None if policy == "base" else LoRARequest(policy, index, str(adapter.resolve()))
        digest = None if policy == "base" else adapter_digest(adapter)
        llm.reset_prefix_cache()
        start = time.monotonic()
        results = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
        seconds = time.monotonic() - start
        records = []
        output_dir = args.output / policy
        output_dir.mkdir()
        for (row, seed), result in zip(assignments, results, strict=True):
            item = cache[row["id"]]
            ids = item["input_ids"].flatten().tolist()
            c = result.outputs[0]
            routes = np.ascontiguousarray(c.routed_experts)
            assert routes.shape == (len(ids) + len(c.token_ids) - 1, 48, 8)
            logprobs = [p[t].logprob for p, t in zip(c.logprobs, c.token_ids, strict=True)]
            assert np.isfinite(logprobs).all()
            raw = tokenizer.decode(c.token_ids, skip_special_tokens=False)
            score = reward(raw, row["answer"], c.finish_reason)
            response = {"prompt_id": row["id"], "policy": policy, "adapter_sha256": digest,
                        "prompt_token_ids": ids, "completion_ids": list(c.token_ids),
                        "completion_logprobs": logprobs, "raw_output": raw, "finish_reason": c.finish_reason,
                        "source_image_sha256": features[row["id"]]["source_image_sha256"],
                        "prompt_sha256": features[row["id"]]["prompt_sha256"],
                        "mm_token_type_ids": item["image_token_mask"].flatten().long().tolist(),
                        "routed_experts": {"data": base64.b64encode(routes.tobytes()).decode(),
                                           "shape": list(routes.shape), "dtype": routes.dtype.name, "start": 0}}
            record = {"request": {"prompt_id": row["id"], "seed": seed, "temperature": .7, "max_tokens": 1536,
                                  "policy": policy}, "response": response, "reward": score, "answer": row["answer"]}
            (output_dir / f"{row['id']}-{seed}.json").write_text(json.dumps(record))
            records.append({"id": row["id"], "seed": seed, "reward": score, "tokens": len(c.token_ids),
                            "finish_reason": c.finish_reason})
        tokens = sum(r["tokens"] for r in records)
        summary[policy] = {"records": records, "correct": sum(r["reward"] for r in records),
                           "truncated": sum(r["finish_reason"] == "length" for r in records),
                           "seconds": seconds, "output_tokens": tokens, "output_tokens_per_second": tokens/seconds}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(policy, {k:v for k,v in summary[policy].items() if k != "records"}, flush=True)


if __name__ == "__main__":
    main()
