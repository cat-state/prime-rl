"""Test frozen reference image embeddings through Prime's vLLM loop backbone."""

import argparse
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--capture-training", action="store_true")
    args = parser.parse_args()
    prompts = []
    for color in ("red", "blue"):
        record = torch.load(args.directory / f"image-reference-{color}-embeds.pt", weights_only=True)
        prompts.append({"prompt_embeds": record["prompt_embeds"]})
    llm = LLM(
        model="/dev/shm/inkling2-loop-bf16",
        tensor_parallel_size=2,
        enable_prompt_embeds=True,
        enforce_eager=True,
        max_model_len=1536,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=4 * 1024**3,
        enable_flashinfer_autotune=False,
        tokenizer_mode="hf",
        aux_output_config={"enable_return_routed_experts": args.capture_training},
    )
    tokenizer = llm.get_tokenizer()
    stop = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    outputs = llm.generate(
        prompts, SamplingParams(temperature=0, max_tokens=1024, stop_token_ids=[stop],
                               logprobs=0 if args.capture_training else None)
    )
    if args.capture_training:
        for color, output in zip(("red", "blue"), outputs, strict=True):
            candidate = output.outputs[0]
            record = torch.load(args.directory / f"image-reference-{color}-embeds.pt", weights_only=True)
            routes = candidate.routed_experts
            if routes is None or routes.ndim != 3 or routes.shape[1] != 48:
                raise ValueError("Missing per-visit image rollout routes")
            logprobs = torch.tensor([row[token].logprob for row, token in
                                    zip(candidate.logprobs, candidate.token_ids, strict=True)])
            if not torch.isfinite(logprobs).all():
                raise ValueError("Nonfinite image rollout logprobs")
            prompt_length = record["prompt_embeds"].shape[0]
            expected = prompt_length + len(candidate.token_ids) - 1
            if routes.shape[0] != expected:
                raise ValueError(f"Image route coverage {routes.shape[0]} != {expected}")
            torch.save({"input_ids": record["input_ids"],
                        "completion_ids": torch.tensor(candidate.token_ids),
                        "completion_logprobs": logprobs,
                        "routed_experts": torch.from_numpy(routes.copy()),
                        "route_start": 0}, args.directory / f"image-rollout-{color}.pt")
    results = [
        {"input_color": color, "output": item.outputs[0].text, "tokens": len(item.outputs[0].token_ids)}
        for color, item in zip(("red", "blue"), outputs, strict=True)
    ]
    name = "image-vllm-training-capture.json" if args.capture_training else "image-vllm-embeds.json"
    (args.directory / name).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
