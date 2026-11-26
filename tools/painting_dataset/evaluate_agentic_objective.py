"""Read-only matched-mode base/painting-adapter replay objective acceptance."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor, distribute_tensor

from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, LoRAConfig, ModelConfig
from prime_rl.trainer.lora import get_lora_state
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger
from tools.painting_dataset.agentic_batch import turn_to_microbatch
from tools.painting_dataset.serve import adapter_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    args = parser.parse_args()
    destination = args.training / "matched-objective.json"
    if destination.exists():
        raise ValueError("Preserve completed acceptance evidence")
    plan = json.loads((args.training / "plan.json").read_text())
    adapter = args.training / "adapter-painting-step1"
    setup_logger("info")
    setup_torch_distributed()
    torch.manual_seed(42)
    config = ModelConfig(
        name="/dev/shm/inkling2-loop-bf16",
        conversion_dir=Path("/dev/shm/inkling2-training"),
        seq_len=max(plan["sequence_lengths"]),
        compile=CompileConfig(fullgraph=False),
        ac=ActivationCheckpointConfig(mode="full", freq=1),
        ep=1,
        cp=1,
        offload="none",
        lora=LoRAConfig(rank=8, alpha=16, target_modules=["o_proj"]),
    )
    model = setup_model(config, get_parallel_dims(config), False)
    get_lora_state().reset_adapter_parameters()
    model.train()
    torch.set_grad_enabled(False)
    samples = [(e, t) for e in plan["trajectories"] for t in e["turns"] if e["advantage"] != 0]
    samples.sort(key=lambda pair: pair[1]["sequence_tokens"], reverse=True)
    evidence = {
        "mode": "compiled train-mode/no_grad for BOTH base and updated",
        "adapter_sha256": adapter_digest(adapter),
        "objective": plan["objective"],
        "policies": {},
    }
    for policy in ("base", "updated"):
        if policy == "updated":
            exported = load_file(str(adapter / "adapter_model.safetensors"))
            targets = get_lora_state().adapter_state_dict()
            if exported.keys() != targets.keys():
                raise ValueError("Adapter key mismatch")
            seen = set()
            for name, target in targets.items():
                if id(target) in seen:
                    continue
                seen.add(id(target))
                value = exported[name].to(device="cuda", dtype=target.dtype)
                if isinstance(target, DTensor):
                    value = distribute_tensor(value, target.device_mesh, target.placements)
                target.copy_(value)
        start = time.monotonic()
        rows = []
        for trajectory, turn in samples:
            directory = Path(turn["directory"])
            encoded = json.loads((directory / "encoded.json").read_text())
            response = json.loads((directory / "response.json").read_text())
            cache = FrozenImageCache(Path(encoded["feature_manifest"]), model.config.hidden_size)
            batch = turn_to_microbatch(response, encoded, advantage=trajectory["advantage"], expected_policy="base")
            tensors = prepare_micro_batch(DataLoader._micro_batch_to_tensor(None, batch), None, cache.materialize)
            ids = tensors["input_ids"].cuda()
            result = model(
                input_ids=ids,
                position_ids=tensors["position_ids"].cuda(),
                seq_lens=tensors["seq_lens"].cuda(),
                labels=ids.roll(-1, dims=-1),
                temperature=tensors["temperatures"].cuda(),
                routed_experts=tensors["routed_experts"].long().cuda().contiguous(),
                **{k: v.cuda() for k, v in tensors["mm_kwargs"].items()},
            )
            mask = tensors["loss_mask"].roll(-1, dims=-1).cuda()
            mask[:, -1] = False
            current = result["logprobs"][mask].float()
            old = tensors["inference_logprobs"].roll(-1, dims=-1).cuda()[mask].float()
            delta = current - old
            ratio = delta.exp()
            advantage = trajectory["advantage"]
            objective = (
                torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).sum()
                * trajectory["loss_coefficient_per_token"]
            )
            if not torch.isfinite(objective) or not torch.isfinite(current).all():
                raise ValueError("Nonfinite replay")
            rows.append(
                {
                    "turn": str(directory),
                    "objective": objective.item(),
                    "logprobs": current.cpu().tolist(),
                    "ratio_min": ratio.min().item(),
                    "ratio_max": ratio.max().item(),
                    "sampled_k3_vs_rollout": (ratio - 1 - delta).mean().item(),
                }
            )
            if dist.get_rank() == 0:
                (args.training / f"matched-{policy}-progress.json").write_text(json.dumps(rows))
            del result, current, old, ratio, delta, objective, tensors
        evidence["policies"][policy] = rows
        evidence[policy + "_seconds"] = time.monotonic() - start
        evidence[policy + "_objective"] = sum(r["objective"] for r in rows)
    evidence["objective_improvement"] = evidence["updated_objective"] - evidence["base_objective"]
    if dist.get_rank() == 0:
        destination.write_text(json.dumps(evidence))
        print({k: v for k, v in evidence.items() if k != "policies"}, flush=True)
    from torch._dynamo.utils import counters

    (args.training / f"matched-compiler-rank{dist.get_rank()}.json").write_text(
        json.dumps({str(k): dict(v) for k, v in counters.items()}, indent=2)
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
