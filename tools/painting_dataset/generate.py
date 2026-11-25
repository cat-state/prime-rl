"""Generate initial shader candidates from cached image-conditioned embeddings."""

import argparse
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    rows = json.loads((args.dataset / "features/manifest.json").read_text())["records"][: args.limit]
    prompts = [{"prompt_embeds": torch.load(row["features"], weights_only=True)["prompt_embeds"]} for row in rows]
    llm = LLM(
        model="/dev/shm/inkling2-loop-bf16",
        tensor_parallel_size=2,
        enable_prompt_embeds=True,
        enforce_eager=True,
        max_model_len=3072,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=6 * 1024**3,
        enable_flashinfer_autotune=False,
        tokenizer_mode="hf",
    )
    tokenizer = llm.get_tokenizer()
    stop = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=2048, stop_token_ids=[stop]))
    destination = args.dataset / "baseline-candidates"
    destination.mkdir(exist_ok=True)
    for row, output in zip(rows, outputs, strict=True):
        candidate = output.outputs[0]
        result = {
            "id": row["id"],
            "split": row["split"],
            "finish_reason": candidate.finish_reason,
            "tokens": len(candidate.token_ids),
            "raw_output": tokenizer.decode(candidate.token_ids, skip_special_tokens=False),
            "validation": "unvalidated; helper/palette prompt and renderer integration pending",
        }
        (destination / f"{row['id']}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "raw_output"}), flush=True)


if __name__ == "__main__":
    main()
