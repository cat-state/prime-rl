"""Fixed-token inference benchmark using the existing frozen image prompts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    items = [torch.load(row["features"], map_location="cpu", weights_only=True)
             for row in manifest["records"][:16]]
    llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=args.tp,
              enable_prompt_embeds=True, enforce_eager=not args.graphs, max_num_seqs=16,
              max_model_len=1024, max_num_batched_tokens=4096,
              kv_cache_memory_bytes=12 * 1024**3, enable_prefix_caching=True,
              enable_flashinfer_autotune=False, tokenizer_mode="hf",
              enable_lora=True, max_lora_rank=8, lora_target_modules=["wo_ud"],
              aux_output_config={"enable_return_routed_experts": True})
    params = SamplingParams(temperature=0, seed=42, max_tokens=256,
                            min_tokens=256, ignore_eos=True, logprobs=0)
    for batch in (1, 4, 8, 16):
        prompts = [{"prompt_embeds": item["prompt_embeds"]} for item in items[:batch]]
        for repeat in range(3):
            llm.reset_prefix_cache()
            start = time.time()
            results = llm.generate(prompts, params, use_tqdm=False)
            end = time.time()
            input_tokens = sum(len(item["input_ids"].flatten()) for item in items[:batch])
            output_tokens = sum(len(result.outputs[0].token_ids) for result in results)
            for item, result in zip(items[:batch], results, strict=True):
                completion = result.outputs[0]
                expected = len(item["input_ids"].flatten()) + len(completion.token_ids) - 1
                assert completion.routed_experts.shape[:2] == (expected, 48)
                assert len(completion.token_ids) == 256
                assert np.isfinite([p[t].logprob for p, t in
                                   zip(completion.logprobs, completion.token_ids, strict=True)]).all()
            row = dict(tp=args.tp, graphs=args.graphs, batch=batch, repeat=repeat,
                       warmup=repeat == 0, start=start, end=end, seconds=end-start,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       output_tokens_per_second=output_tokens/(end-start),
                       input_tokens_per_second_amortized=input_tokens/(end-start))
            (args.output / f"batch{batch}-repeat{repeat}.json").write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
