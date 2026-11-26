"""Barrier-synchronized DP/TP graph benchmark on identical cached prompt sets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams


def barrier(directory, label, rank, replicas):
    if replicas == 1:
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}.{rank}.ready").touch()
    deadline = time.monotonic() + 600
    while not all((directory / f"{label}.{r}.ready").exists() for r in range(replicas)):
        if time.monotonic() > deadline:
            raise TimeoutError(f"Barrier {label} waiting for replicas")
        time.sleep(0.005)
    release = directory / f"{label}.release"
    if rank == 0:
        temporary = release.with_suffix(".tmp")
        temporary.write_text(str(time.time() + 0.25))
        temporary.rename(release)
    while not release.exists():
        time.sleep(0.001)
    target = float(release.read_text())
    while time.time() < target:
        time.sleep(min(0.001, max(0, target - time.time())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--barrier-dir", type=Path)
    parser.add_argument("--total-batches", type=int, nargs="+", default=[2, 8, 16, 32])
    parser.add_argument("--kv-gib", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    items = [torch.load(row["features"], map_location="cpu", weights_only=True)
             for row in manifest["records"][:max(args.total_batches)]]
    items = [items[i % len(items)] for i in range(max(args.total_batches))]
    llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=args.tp,
              enable_prompt_embeds=True, enforce_eager=not args.graphs,
              max_num_seqs=max(args.total_batches) // args.replicas,
              max_model_len=1024, max_num_batched_tokens=4096,
              kv_cache_memory_bytes=args.kv_gib * 1024**3, enable_prefix_caching=True,
              enable_flashinfer_autotune=False, tokenizer_mode="hf",
              enable_lora=True, max_lora_rank=8, lora_target_modules=["wo_ud"],
              aux_output_config={"enable_return_routed_experts": True})
    params = SamplingParams(temperature=0, seed=42, max_tokens=256,
                            min_tokens=256, ignore_eos=True, logprobs=0)
    for total_batch in args.total_batches:
        selected = items[:total_batch][args.rank::args.replicas]
        batch = len(selected)
        prompts = [{"prompt_embeds": item["prompt_embeds"]} for item in selected]
        for repeat in range(3):
            llm.reset_prefix_cache()
            barrier(args.barrier_dir, f"batch{total_batch}-repeat{repeat}", args.rank, args.replicas)
            start = time.time()
            results = llm.generate(prompts, params, use_tqdm=False)
            end = time.time()
            input_tokens = sum(len(item["input_ids"].flatten()) for item in selected)
            output_tokens = sum(len(result.outputs[0].token_ids) for result in results)
            for item, result in zip(selected, results, strict=True):
                completion = result.outputs[0]
                expected = len(item["input_ids"].flatten()) + len(completion.token_ids) - 1
                assert completion.routed_experts.shape[:2] == (expected, 48)
                assert len(completion.token_ids) == 256
                assert np.isfinite([p[t].logprob for p, t in
                                   zip(completion.logprobs, completion.token_ids, strict=True)]).all()
            row = dict(tp=args.tp, graphs=args.graphs, batch=batch, total_batch=total_batch,
                       kv_gib_per_gpu=args.kv_gib,
                       rank=args.rank, replicas=args.replicas, repeat=repeat,
                       warmup=repeat == 0, start=start, end=end, seconds=end-start,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       output_tokens_per_second=output_tokens/(end-start),
                       input_tokens_per_second_amortized=input_tokens/(end-start))
            (args.output / f"batch{batch}-repeat{repeat}.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
