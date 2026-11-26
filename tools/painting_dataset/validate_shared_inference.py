"""Validate shared loop experts, base/active-LoRA inference, and fixed-token batching."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def worker_report(worker):
    from prime_rl.inference.vllm.models.inkling2.shared_experts import storage_report
    model = worker.model_runner.get_model()
    groups = model.model.shared_expert_groups
    visits = model.model.layers
    return {"groups": storage_report(groups),
            "unique_attention_modules": len({id(layer.attn) for layer in visits}),
            "unique_conv_states": len({id(layer.conv_state) for layer in visits}),
            "unique_lora_wrappers": len({id(layer.attn.wo_ud) for layer in visits}),
            "allocated_gib": torch.cuda.memory_allocated()/2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}


def record_result(item, result, expected_tokens):
    completion = result.outputs[0]
    ids = list(completion.token_ids)
    logprobs = [p[t].logprob for p, t in zip(completion.logprobs, ids, strict=True)]
    routes = completion.routed_experts
    assert len(ids) == expected_tokens and np.isfinite(logprobs).all()
    assert routes.shape == (len(item["input_ids"].flatten()) + len(ids) - 1, 48, 8)
    return {"ids": ids, "logprobs": logprobs, "routes_shape": list(routes.shape),
            "routes_sha256": hashlib.sha256(routes.tobytes()).hexdigest(), "raw_output": completion.text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.manifest.read_text())["records"][:16]
    items = [torch.load(row["features"], map_location="cpu", weights_only=True) for row in rows]
    llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=args.tp,
              enable_prompt_embeds=True, enforce_eager=not args.graphs, max_num_seqs=16,
              max_model_len=1024, max_num_batched_tokens=4096,
              kv_cache_memory_bytes=12 * 1024**3, enable_prefix_caching=True,
              enable_flashinfer_autotune=False, tokenizer_mode="hf",
              enable_lora=True, max_lora_rank=8, lora_target_modules=["wo_ud"],
              kernel_config={"moe_backend": "triton"},
              aux_output_config={"enable_return_routed_experts": True})
    reports = llm.collective_rpc(worker_report)
    (args.output / "storage.json").write_text(json.dumps(reports, indent=2))
    for report in reports:
        assert report["unique_attention_modules"] == report["unique_conv_states"] == report["unique_lora_wrappers"] == 48
    prompts = [{"prompt_embeds": item["prompt_embeds"]} for item in items[:4]]
    params = SamplingParams(temperature=0, seed=42, max_tokens=64, min_tokens=64, ignore_eos=True, logprobs=0)
    adapter = LoRARequest("counting-eager-step1", 1, str(args.adapter.resolve()))
    for policy, request in (("base-before", None), ("adapter", adapter), ("base-after", None)):
        llm.reset_prefix_cache()
        outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
        evidence = [record_result(item, result, 64) for item, result in zip(items, outputs)]
        (args.output / f"{policy}.json").write_text(json.dumps(evidence, indent=2))
    if args.reload:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": "/dev/shm/inkling2-loop-bf16"})
        reports = llm.collective_rpc(worker_report)
        (args.output / "storage-after-reload.json").write_text(json.dumps(reports, indent=2))
        adapter = LoRARequest("counting-eager-reloaded", 2, str(args.adapter.resolve()))
        for policy, request in (("reloaded-base", None), ("reloaded-adapter", adapter)):
            llm.reset_prefix_cache()
            outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
            evidence = [record_result(item, result, 64) for item, result in zip(items, outputs)]
            (args.output / f"{policy}.json").write_text(json.dumps(evidence, indent=2))
    if args.benchmark:
        params = SamplingParams(temperature=0, seed=42, max_tokens=256, min_tokens=256, ignore_eos=True, logprobs=0)
        for batch in (1, 4, 8, 16):
            prompts = [{"prompt_embeds": item["prompt_embeds"]} for item in items[:batch]]
            for repeat in range(3):
                llm.reset_prefix_cache()
                start = time.time()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                end = time.time()
                for item, result in zip(items[:batch], outputs, strict=True):
                    record_result(item, result, 256)
                row = dict(tp=args.tp, graphs=args.graphs, batch=batch, repeat=repeat, warmup=repeat == 0,
                           start=start, end=end, seconds=end-start, output_tokens=batch*256,
                           output_tokens_per_second=batch*256/(end-start))
                (args.output / f"batch{batch}-repeat{repeat}.json").write_text(json.dumps(row, indent=2))
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
