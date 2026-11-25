"""Test frozen reference image embeddings through Prime's vLLM loop backbone."""

import argparse
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
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
    )
    tokenizer = llm.get_tokenizer()
    stop = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1024, stop_token_ids=[stop]))
    results = [
        {"input_color": color, "output": item.outputs[0].text, "tokens": len(item.outputs[0].token_ids)}
        for color, item in zip(("red", "blue"), outputs, strict=True)
    ]
    (args.directory / "image-vllm-embeds.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
