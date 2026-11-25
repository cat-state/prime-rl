"""Full-model teacher-forced image rollout comparison with captured per-visit routes."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger


def main():
    setup_logger("info")
    setup_torch_distributed()
    config = ModelConfig(name="/dev/shm/inkling2-loop-bf16", conversion_dir=Path("/dev/shm/inkling2-training"),
                         compile=None, ep=1, cp=1, offload="none")
    model = setup_model(config, get_parallel_dims(config), False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.name)
    image_token = tokenizer.convert_tokens_to_ids("<|unused_200054|>")
    results = []
    root = Path("loop-validation")
    for color in ("red", "blue"):
        rollout = torch.load(root / f"image-rollout-{color}.pt", weights_only=True)
        reference = torch.load(root / f"image-reference-{color}-embeds.pt", weights_only=True)
        prompt = rollout["input_ids"].flatten().cuda()
        completion = rollout["completion_ids"].cuda()
        ids = torch.cat((prompt, completion)).unsqueeze(0)
        with torch.no_grad():
            # Use the normal root forward so FSDP gathers embedding parameters.
            image_mask = ids[:, :-1] == image_token
            features = reference["prompt_embeds"][prompt.cpu() == image_token].cuda()
            if not len(features):
                raise ValueError("Reference prompt has no image placeholders")
            length = ids.shape[1] - 1
            output = model(input_ids=ids[:, :-1], image_features=features,
                           image_token_mask=image_mask, mm_token_type_ids=image_mask.long(),
                           position_ids=torch.arange(length, device="cuda").unsqueeze(0),
                           seq_lens=torch.tensor([length], device="cuda"), labels=ids[:, 1:],
                           temperature=torch.ones((1, length), device="cuda"),
                           routed_experts=rollout["routed_experts"].long().cuda().unsqueeze(0).contiguous())
        actual = output["logprobs"].flatten()[prompt.numel()-1:].float().cpu()
        expected = rollout["completion_logprobs"].float()
        difference = actual - expected
        results.append({"color": color, "tokens": len(expected), "finite": bool(torch.isfinite(actual).all()),
                        "max_abs_logprob_error": difference.abs().max().item(),
                        "mean_abs_logprob_error": difference.abs().mean().item(),
                        "sampled_k3": (torch.exp(difference)-difference-1).mean().item()})
        if dist.get_rank() == 0:
            (root / "image-trainer-parity.json").write_text(json.dumps(results, indent=2))
            print(results[-1], flush=True)
    # Diagnostic evidence first: no new tolerance chosen after seeing results.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
