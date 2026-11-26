"""Real frozen-image encoder and shared-weight vLLM episode providers."""
import base64
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tools.painting_dataset.serve import adapter_digest


class FrozenEncoderClient:
    def __init__(self, workspace: Path, gpu: str = "3", final_only: bool = False):
        self.workspace = workspace.resolve()
        self.gpu = gpu
        self.final_only = final_only
        self.socket = self.workspace / "encoder.sock"
        if len(os.fsencode(self.socket)) >= 108:
            raise ValueError("Encoder socket path too long")
        self.process = None
        self.log = None

    def __enter__(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log = (self.workspace / "encoder.log").open("w")
        command = ["/root/.local/bin/uv", "run", "--project", "/root/inkling-mini", "--no-sync", "python",
                   str(Path(__file__).with_name("agentic_encode_worker.py")), "--socket", str(self.socket),
                   "--workspace", str(self.workspace)]
        self.process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu),
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 600
        while not self.socket.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.__exit__()
                raise RuntimeError("Encoder startup failed; inspect encoder.log")
            time.sleep(.2)
        return self

    def __call__(self, messages, directory):
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(300)
            client.connect(str(self.socket))
            client.sendall((json.dumps({"messages": messages, "directory": str(directory),
                                       "final_only": self.final_only}) + "\n").encode())
            result = json.loads(client.makefile("rb").readline(8_000_000))
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["encoded"]

    def __exit__(self, *args):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log is not None:
            self.log.close()
        self.socket.unlink(missing_ok=True)


class SharedInferenceGenerator:
    def __init__(self, *, max_model_len=6144, adapter: Path | None = None, policy="base"):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        self.llm = LLM(model="/dev/shm/inkling2-loop-bf16", tensor_parallel_size=1,
                       enable_prompt_embeds=True, enforce_eager=False, max_num_seqs=1,
                       max_model_len=max_model_len, max_num_batched_tokens=4096,
                       kv_cache_memory_bytes=24 * 1024**3, enable_prefix_caching=True,
                       enable_flashinfer_autotune=False, tokenizer_mode="hf", enable_lora=True,
                       max_lora_rank=8, lora_target_modules=["wo_ud"],
                       aux_output_config={"enable_return_routed_experts": True})
        self.tokenizer = self.llm.get_tokenizer()
        self.stop = self.tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")
        self.policy = policy
        self.adapter = adapter
        self.digest = adapter_digest(adapter) if adapter else None
        self.lora = LoRARequest(policy, 1, str(adapter.resolve())) if adapter else None

    def __call__(self, encoded, request):
        from vllm import SamplingParams
        if request["policy"] != self.policy:
            raise ValueError("Generator policy mismatch")
        if self.adapter and adapter_digest(self.adapter) != self.digest:
            raise ValueError("Adapter changed")
        item = torch.load(encoded["features"], weights_only=True, map_location="cpu")
        if item["input_ids"].flatten().tolist() != encoded["input_ids"]:
            raise ValueError("Encoded feature tokens changed")
        self.llm.reset_prefix_cache()
        start = time.monotonic()
        completion = self.llm.generate([{"prompt_embeds": item["prompt_embeds"]}],
            SamplingParams(temperature=request["temperature"], seed=request["seed"], max_tokens=request["max_tokens"],
                           logprobs=0, stop_token_ids=[self.stop]), lora_request=self.lora, use_tqdm=False)[0].outputs[0]
        seconds = time.monotonic() - start
        routes = np.ascontiguousarray(completion.routed_experts)
        logprobs = [p[t].logprob for p, t in zip(completion.logprobs, completion.token_ids, strict=True)]
        return {"policy": self.policy, "adapter_sha256": self.digest, "temperature": request["temperature"],
                "seed": request["seed"], "prompt_token_ids": encoded["input_ids"],
                "completion_ids": list(completion.token_ids), "completion_logprobs": logprobs,
                "finish_reason": completion.finish_reason,
                "raw_output": self.tokenizer.decode(completion.token_ids, skip_special_tokens=False),
                "generation_seconds": seconds,
                "routed_experts": {"start": 0, "shape": list(routes.shape), "dtype": routes.dtype.name,
                                   "data": base64.b64encode(routes.tobytes()).decode()}}
