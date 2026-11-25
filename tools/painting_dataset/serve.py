"""Local, single-turn frozen-prompt rollout endpoint; not a native image API.

Only manifest-listed prompts are accepted. Cached text embeddings require frozen
base embedding weights. Use a custom client; the stock Prime renderer is text-only.
"""

import argparse
import base64
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class Request(BaseModel):
    prompt_id: str
    policy: str = "base"
    max_tokens: int = Field(128, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0, le=2)
    seed: int = Field(0, ge=0)


class Adapter(BaseModel):
    policy: str
    path: str


def adapter_digest(path):
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("encoder_frozen") is not True:
        raise ValueError("Requires frozen encoder cache")
    records = {row["id"]: row for row in manifest["records"]}
    cache = {}
    for key, row in records.items():
        item = torch.load(row["features"], weights_only=True, map_location="cpu")
        ids, mask, embeds = item["input_ids"].flatten(), item["image_token_mask"].flatten(), item["prompt_embeds"]
        if len(ids) != len(mask) or len(ids) != len(embeds) or not torch.isfinite(embeds).all():
            raise ValueError(f"Invalid cached prompt: {key}")
        cache[key] = item
    llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=2,
              enable_prompt_embeds=True, enforce_eager=True, max_num_seqs=1,
              max_model_len=max(len(v["input_ids"].flatten()) for v in cache.values()) + 8192,
              max_num_batched_tokens=1024, kv_cache_memory_bytes=12 * 1024**3,
              enable_flashinfer_autotune=False, tokenizer_mode="hf",
              enable_lora=True, max_lora_rank=8, lora_target_modules=["wo_ud"],
              aux_output_config={"enable_return_routed_experts": True})
    tokenizer = llm.get_tokenizer()
    stop = tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
    lock = threading.Lock()
    adapters = {}
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ready": True, "prompts": list(cache), "policies": ["base", *adapters], "single_turn": True}

    @app.post("/adapters")
    def register(adapter: Adapter):
        with lock:
            if not adapter.policy or adapter.policy == "base" or adapter.policy in adapters:
                raise HTTPException(409, "Policy name must be new and cannot be base")
            path = Path(adapter.path).resolve()
            try:
                config = json.loads((path / "adapter_config.json").read_text())
                if config["r"] > 8 or config["target_modules"] != ["wo_ud"]:
                    raise ValueError("Requires rank<=8 wo_ud adapter")
                digest = adapter_digest(path)
            except (OSError, KeyError, ValueError) as error:
                raise HTTPException(400, str(error)) from error
            adapters[adapter.policy] = (LoRARequest(adapter.policy, len(adapters) + 1, str(path)), digest)
            return {"policy": adapter.policy, "sha256": digest, "loaded": False}

    @app.post("/generate")
    def generate(request: Request):
        if request.prompt_id not in cache:
            raise HTTPException(404, "Unknown cached prompt")
        item = cache[request.prompt_id]
        with lock:
            lora, digest = None, None
            if request.policy != "base":
                if request.policy not in adapters:
                    raise HTTPException(404, "Unknown adapter policy")
                lora, digest = adapters[request.policy]
                if adapter_digest(Path(lora.lora_path)) != digest:
                    raise HTTPException(409, "Registered adapter files changed; register a new policy")
            result = llm.generate([{"prompt_embeds": item["prompt_embeds"]}],
                                  SamplingParams(temperature=request.temperature, seed=request.seed,
                                                 max_tokens=request.max_tokens, logprobs=0,
                                                 stop_token_ids=[stop]), use_tqdm=False,
                                  lora_request=lora)[0].outputs[0]
        ids = item["input_ids"].flatten().tolist()
        routes = result.routed_experts
        if routes is None or routes.shape[:2] != (len(ids) + len(result.token_ids) - 1, 48):
            raise RuntimeError("Incomplete per-visit route coverage")
        logprobs = [row[token].logprob for row, token in zip(result.logprobs, result.token_ids, strict=True)]
        if not np.isfinite(logprobs).all():
            raise RuntimeError("Nonfinite rollout logprobs")
        routes = np.ascontiguousarray(routes)
        return {"prompt_id": request.prompt_id, "policy": request.policy, "adapter_sha256": digest,
                "prompt_token_ids": ids,
                "source_image_sha256": records[request.prompt_id]["source_image_sha256"],
                "prompt_sha256": records[request.prompt_id]["prompt_sha256"],
                "mm_token_type_ids": item["image_token_mask"].flatten().long().tolist(),
                "completion_ids": list(result.token_ids), "completion_logprobs": logprobs,
                "finish_reason": result.finish_reason,
                "raw_output": tokenizer.decode(result.token_ids, skip_special_tokens=False),
                "routed_experts": {"data": base64.b64encode(routes.tobytes()).decode(),
                                   "shape": list(routes.shape), "dtype": routes.dtype.name, "start": 0}}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
