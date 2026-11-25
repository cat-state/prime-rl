"""Full-model teacher-forced image rollout comparison with captured per-visit routes."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from prime_rl.configs.trainer import CompileConfig, LoRAConfig, ModelConfig
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-update", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    setup_logger("info")
    setup_torch_distributed()
    config = ModelConfig(name="/dev/shm/inkling2-loop-bf16", conversion_dir=Path("/dev/shm/inkling2-training"),
                         compile=CompileConfig(fullgraph=False) if args.compile else None,
                         lora=LoRAConfig(rank=8, alpha=16, target_modules=["o_proj"]) if args.lora_update else None,
                         ep=1, cp=1, offload="none")
    model = setup_model(config, get_parallel_dims(config), False)
    if args.lora_update:
        from prime_rl.trainer.lora import get_lora_state

        # Match the real trainer: base checkpoint loading leaves adapters to this reset.
        torch.manual_seed(42)
        get_lora_state().reset_adapter_parameters()
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5) if args.lora_update else None
    if optimizer:
        model.train()
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
        with torch.enable_grad() if args.lora_update else torch.no_grad():
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
        sampled = output["logprobs"].flatten()[prompt.numel()-1:]
        actual = sampled.detach().float().cpu()
        expected = rollout["completion_logprobs"].float()
        difference = actual - expected
        results.append({"color": color, "tokens": len(expected), "finite": bool(torch.isfinite(actual).all()),
                        "max_abs_logprob_error": difference.abs().max().item(),
                        "mean_abs_logprob_error": difference.abs().mean().item(),
                        "sampled_k3": (torch.exp(difference)-difference-1).mean().item()})
        if optimizer:
            # Fixed positive advantage is an update-plumbing smoke, not environment RL.
            ratio = torch.exp(sampled - expected.to(sampled))
            loss = -torch.minimum(ratio, ratio.clamp(0.8, 1.2)).mean()
            loss.backward()
            def local(tensor):
                return tensor.to_local() if hasattr(tensor, "to_local") else tensor
            norm2 = sum(local(p.grad).float().square().sum() for p in trainable if p.grad is not None)
            dist.all_reduce(norm2)
            if not torch.isfinite(norm2) or norm2 <= 0:
                diagnostic = {"rank": dist.get_rank(), "loss": loss.item(), "gradient_norm_squared": norm2.item(),
                              "ratio_min": ratio.min().item(), "ratio_max": ratio.max().item(),
                              "parameters_with_grad": sum(p.grad is not None for p in trainable),
                              "nonfinite_gradients": [name for name, p in model.named_parameters()
                                                      if p.grad is not None and not torch.isfinite(local(p.grad)).all()]}
                (root / f"image-lora-gradient-failure-rank{dist.get_rank()}.json").write_text(json.dumps(diagnostic, indent=2))
                raise ValueError("Image LoRA gradient must be finite and nonzero")
            before = [local(p).detach().clone() for p in trainable]
            optimizer.step()
            delta2 = sum((local(p).detach() - old).float().square().sum()
                         for p, old in zip(trainable, before, strict=True))
            dist.all_reduce(delta2)
            if not torch.isfinite(delta2) or delta2 <= 0:
                raise ValueError("Image LoRA update must be finite and nonzero")
            optimizer.zero_grad(set_to_none=True)
            results[-1].update(loss=loss.item(), gradient_norm=norm2.sqrt().item(),
                               adapter_delta_norm=delta2.sqrt().item(),
                               objective="fixed positive advantage smoke; not scored environment RL")
        if dist.get_rank() == 0:
            name = "image-lora-update" if args.lora_update else "image-trainer-parity"
            name += "-compiled" if args.compile else ""
            (root / f"{name}.json").write_text(json.dumps(results, indent=2))
            print(results[-1], flush=True)
    # Diagnostic evidence first: no new tolerance chosen after seeing results.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
