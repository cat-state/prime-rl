"""Controlled variable-length art workload; identical prompt/seed workload at each chunk size."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tools.painting_dataset.agentic_episode import extract_action
from tools.painting_dataset.serve import adapter_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    encoded = [json.loads((args.episode / f"turn-{t}/encoded.json").read_text()) for t in (0, 2)]
    features = [torch.load(e["features"], map_location="cpu", weights_only=True)["prompt_embeds"] for e in encoded]
    assignments = [{"prompt": i % 2, "seed": 501 + i} for i in range(16)]
    provenance = {"assignments": assignments, "encoded": encoded, "adapter_sha256": adapter_digest(args.adapter),
                  "note": "Two cached prompts from ONE artwork, cycled; active counting adapter, not trained painting adapter",
                  "max_model_len": 12288, "max_tokens": 8192, "kv_cache_gib_per_gpu": 36}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=2,
              enable_prompt_embeds=True, enforce_eager=False, max_num_seqs=16,
              max_model_len=12288, max_num_batched_tokens=4096,
              kv_cache_memory_bytes=36 * 1024**3, enable_prefix_caching=True,
              enable_flashinfer_autotune=False, tokenizer_mode="hf", enable_lora=True,
              max_lora_rank=8, lora_target_modules=["wo_ud"],
              aux_output_config={"enable_return_routed_experts": True})
    tokenizer = llm.get_tokenizer()
    stop = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    lora = LoRARequest("compiled", 1, str(args.adapter.resolve()))
    # Short warmup establishes adapter graph path; full-length shape costs remain measured.
    llm.generate([{"prompt_embeds": f} for f in features],
                 SamplingParams(temperature=.7, seed=500, max_tokens=16, logprobs=0, stop_token_ids=[stop]),
                 lora_request=lora, use_tqdm=False)
    summaries = []
    for batch in (4, 8, 16):
        directory = args.output / f"batch-{batch}"
        directory.mkdir()
        rows = []
        seconds = 0.
        for start in range(0, len(assignments), batch):
            group = assignments[start:start + batch]
            llm.reset_prefix_cache()
            params = [SamplingParams(temperature=.7, seed=a["seed"], max_tokens=8192, logprobs=0,
                                     stop_token_ids=[stop]) for a in group]
            begin = time.monotonic()
            results = llm.generate([{"prompt_embeds": features[a["prompt"]]} for a in group], params,
                                   lora_request=lora, use_tqdm=False)
            elapsed = time.monotonic() - begin
            seconds += elapsed
            for offset, (assignment, result) in enumerate(zip(group, results, strict=True)):
                c = result.outputs[0]
                raw = tokenizer.decode(c.token_ids, skip_special_tokens=False)
                response = {"raw_output": raw, "finish_reason": c.finish_reason}
                try:
                    source = extract_action(response, "paint")
                    complete_code = True
                    (directory / f"{start+offset}.wgsl").write_text(source)
                except ValueError:
                    complete_code = False
                routes = np.ascontiguousarray(c.routed_experts)
                prompt_ids = encoded[assignment["prompt"]]["input_ids"]
                assert routes.shape == (len(prompt_ids) + len(c.token_ids) - 1, 48, 8)
                logprobs = [p[t].logprob for p, t in zip(c.logprobs, c.token_ids, strict=True)]
                assert np.isfinite(logprobs).all()
                np.save(directory / f"{start+offset}-routes.npy", routes)
                response.update(assignment, completion_ids=list(c.token_ids), completion_logprobs=logprobs,
                                prompt_token_ids=prompt_ids, chunk_seconds=elapsed)
                (directory / f"{start+offset}.json").write_text(json.dumps(response))
                rows.append({**assignment, "tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                             "complete_code": complete_code, "chunk_seconds": elapsed})
            print(json.dumps({"batch": batch, "finished": start + len(group), "chunk_seconds": elapsed}), flush=True)
        tokens = sum(r["tokens"] for r in rows)
        summary = {"batch": batch, "records": rows, "seconds": seconds, "output_tokens": tokens,
                   "tokens_per_second": tokens / seconds,
                   "truncated": sum(r["finish_reason"] == "length" for r in rows),
                   "complete_code": sum(r["complete_code"] for r in rows)}
        summaries.append(summary)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
