"""Scored-image LoRA updates with optional group-centered gradient accumulation."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from prime_rl.configs.trainer import CompileConfig, LoRAConfig, ModelConfig
from prime_rl.trainer.lora import get_lora_state
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger
from tools.painting_dataset.batch import rollout_to_microbatch
from tools.painting_dataset.colors import reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing LoRA update on original fixed rollouts; no optimizer steps")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evidence")
    setup_logger("info")
    setup_torch_distributed()
    torch.manual_seed(42)
    config = ModelConfig(name="/dev/shm/inkling2-loop-bf16", conversion_dir=Path("/dev/shm/inkling2-training"),
                         compile=CompileConfig(fullgraph=False) if args.compile else None, ep=1, cp=1, offload="none",
                         lora=LoRAConfig(rank=8, alpha=16, target_modules=["o_proj"]))
    model = setup_model(config, get_parallel_dims(config), False)
    get_lora_state().reset_adapter_parameters()
    model.train()
    root = args.dataset
    cache = FrozenImageCache(root / "features/manifest.json", model.config.hidden_size)
    samples = []
    for row in json.loads((root / "manifest.json").read_text())["records"]:
        if row["split"] != "train":
            continue
        group = [json.loads((root / "rollouts-base" / f"{row['id']}-{i}.json").read_text()) for i in range(4)]
        scores = [reward(r["response"]["raw_output"], row["answer"], r["response"]["finish_reason"]) for r in group]
        baseline = sum(scores) / len(scores)
        for i, (record, score) in enumerate(zip(group, scores, strict=True)):
            assert score == record["reward"]
            if score != baseline:
                samples.append((row, i, record, score - baseline))
    assert len(samples) == 4
    evidence = {"compiled": args.compile, "adapter": str(args.adapter), "policies": {}}
    for policy in ("base", "updated"):
        if policy == "updated":
            from torch.distributed.tensor import DTensor, distribute_tensor
            exported = load_file(str(args.adapter / "adapter_model.safetensors"))
            targets = get_lora_state().adapter_state_dict()
            assert exported.keys() == targets.keys()
            seen = {}
            with torch.no_grad():
                for name, target in targets.items():
                    source = exported[name]
                    assert torch.isfinite(source).all() and source.shape == target.shape
                    if id(target) in seen:
                        assert torch.equal(source, exported[seen[id(target)]])
                        continue
                    seen[id(target)] = name
                    value = source.to(device="cuda", dtype=target.dtype)
                    if isinstance(target, DTensor):
                        value = distribute_tensor(value, target.device_mesh, target.placements)
                    target.copy_(value)
        metrics = []
        for row, sample_index, record, advantage in samples:
            batch = rollout_to_microbatch(record["response"], Path(row["image"]).read_bytes(),
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
            current = output["logprobs"][mask].detach().float()
            old = tensors["inference_logprobs"].roll(-1, dims=-1).cuda()[mask].float()
            delta = current - old
            ratio = delta.exp()
            objective = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            assert torch.isfinite(current).all() and torch.isfinite(ratio).all()
            metrics.append({"id": row["id"], "sample_index": sample_index, "advantage": advantage,
                            "tokens": current.numel(), "objective": objective.item(),
                            "sampled_k3_vs_rollout": (ratio - 1 - delta).mean().item(),
                            "ratio_min": ratio.min().item(), "ratio_max": ratio.max().item(),
                            "clip_fraction": ((ratio < .8) | (ratio > 1.2)).float().mean().item(),
                            "sampled_logprobs": current.cpu().tolist()})
            del output, current, ratio, objective
            if dist.get_rank() == 0:
                print(policy, {k:v for k,v in metrics[-1].items() if k != "sampled_logprobs"}, flush=True)
        evidence["policies"][policy] = metrics
        evidence[policy + "_mean_objective"] = sum(m["objective"] for m in metrics)/len(metrics)
    evidence["objective_improvement"] = evidence["updated_mean_objective"] - evidence["base_mean_objective"]
    from torch._dynamo.utils import counters
    evidence["compiler_counters"] = {str(g): {str(k):v for k,v in c.items()} for g,c in counters.items()}
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(evidence, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
