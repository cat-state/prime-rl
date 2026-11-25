"""Four scored-image LoRA updates using cached on-policy HTTP rollouts.

Uses reward as a zero-baseline advantage, not group-centered GRPO. This small
integration test cannot demonstrate generalization when baseline accuracy is 1.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.lora import get_lora_state, save_lora_config
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger
from tools.painting_dataset.batch import rollout_to_microbatch
from tools.painting_dataset.colors import reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("loop-validation/color-task"))
    parser.add_argument("--group-centered", action="store_true", help="Accumulate one update from four samples per prompt")
    args = parser.parse_args()
    setup_logger("info")
    setup_torch_distributed()
    torch.manual_seed(42)
    root = args.dataset
    destination = root / ("adapter-group-step1" if args.group_centered else "adapter-step4")
    if destination.exists():
        raise ValueError("Refusing to overwrite prior adapter")
    config = ModelConfig(name="/dev/shm/inkling2-loop-bf16", conversion_dir=Path("/dev/shm/inkling2-training"),
                         compile=None, ep=1, cp=1, offload="none",
                         lora=LoRAConfig(rank=8, alpha=16, target_modules=["o_proj"]))
    model = setup_model(config, get_parallel_dims(config), False)
    get_lora_state().reset_adapter_parameters()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0)
    cache = FrozenImageCache(root / "features/manifest.json", model.config.hidden_size)
    metrics = []
    rows = json.loads((root / "manifest.json").read_text())["records"]
    samples = []
    for row in (r for r in rows if r["split"] == "train"):
        group = [json.loads((root / "rollouts-base" / f"{row['id']}-{i}.json").read_text())
                 for i in range(4 if args.group_centered else 1)]
        scores = [reward(r["response"]["raw_output"], row["answer"], r["response"]["finish_reason"]) for r in group]
        baseline = sum(scores) / len(scores) if args.group_centered else 0
        for record, score in zip(group, scores, strict=True):
            assert score == record["reward"]
            if score != baseline:
                samples.append((row, record, score - baseline))
    if not samples:
        raise ValueError("No nonzero advantages; refusing an empty update")
    for index, (row, record, advantage) in enumerate(samples):
        response = record["response"]
        score = reward(response["raw_output"], row["answer"], response["finish_reason"])
        assert score == record["reward"]
        batch = rollout_to_microbatch(response, Path(row["image"]).read_bytes(),
                                      temperature=record["request"]["temperature"], advantage=advantage, expected_policy="base")
        tensors = prepare_micro_batch(DataLoader._micro_batch_to_tensor(None, batch), None, cache.materialize)
        ids = tensors["input_ids"].cuda()
        output = model(input_ids=ids, position_ids=tensors["position_ids"].cuda(),
                       seq_lens=tensors["seq_lens"].cuda(), labels=ids.roll(-1, dims=-1),
                       temperature=tensors["temperatures"].cuda(),
                       routed_experts=tensors["routed_experts"].long().cuda().contiguous(),
                       **{k: v.cuda() for k, v in tensors["mm_kwargs"].items()})
        mask = tensors["loss_mask"].roll(-1, dims=-1).cuda()
        mask[:, -1] = False
        old = tensors["inference_logprobs"].roll(-1, dims=-1).cuda()
        ratio = (output["logprobs"][mask] - old[mask]).exp()
        loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
        if args.group_centered:
            loss = loss / len(samples)
        loss.backward()
        def local(t):
            return t.to_local() if hasattr(t, "to_local") else t
        norm = sum(local(p.grad).float().square().sum() for p in params if p.grad is not None)
        dist.all_reduce(norm)
        if not torch.isfinite(norm) or (score > 0 and norm <= 0):
            raise ValueError("Invalid scored image gradients")
        if not args.group_centered or index == len(samples) - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.append({"id": row["id"], "reward": score, "advantage": advantage,
                        "loss": loss.item(), "gradient_norm": norm.sqrt().item()})
        if dist.get_rank() == 0:
            (root / "scored-update-metrics.json").write_text(json.dumps(metrics, indent=2))
            print(metrics[-1], flush=True)
    state = get_lora_state().adapter_state_dict()
    full = {k: (v.full_tensor() if hasattr(v, "full_tensor") else v).detach().cpu().contiguous().clone()
            for k, v in state.items()}
    if not all(torch.isfinite(v).all() for v in full.values()):
        raise ValueError("Nonfinite exported adapter")
    if dist.get_rank() == 0:
        destination.mkdir()
        save_file(full, str(destination / "adapter_model.safetensors"))
        save_lora_config(model, destination, rank=8, alpha=16, dropout=0)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
