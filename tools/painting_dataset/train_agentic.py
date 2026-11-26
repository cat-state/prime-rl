"""Plan or execute one trajectory-balanced compiled LoRA update from saved painting episodes."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def build_plan(root, rewards=None, learning_rate=1e-5):
    if not 0 < learning_rate < float("inf"):
        raise ValueError("Learning rate must be finite and positive")
    timing = json.loads((root / "timing.json").read_text())
    jobs = json.loads((root / "jobs.json").read_text())
    reward_rows = json.loads(rewards.read_text()) if rewards else None
    trajectories = []
    for job, path in zip(jobs, timing["episodes"], strict=True):
        episode = Path(path)
        state = json.loads((episode / "episode.json").read_text())
        if not state["complete"] or state["policy"] != "base" or state["seed"] != job["seed"]:
            raise ValueError("Incomplete or mismatched base trajectory")
        turns = []
        for index in range(len(state["turns"])):
            directory = episode / f"turn-{index}"
            response = json.loads((directory / "response.json").read_text())
            turns.append(
                {
                    "directory": str(directory),
                    "prompt_tokens": len(response["prompt_token_ids"]),
                    "completion_tokens": len(response["completion_ids"]),
                    "sequence_tokens": len(response["prompt_token_ids"]) + len(response["completion_ids"]),
                }
            )
        trajectories.append(
            {
                "episode": str(episode),
                "artwork_id": job["artwork_id"],
                "seed": job["seed"],
                "reward": reward_rows[str(episode)] if reward_rows else None,
                "completion_tokens": sum(t["completion_tokens"] for t in turns),
                "turns": turns,
            }
        )
    groups = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory["artwork_id"], []).append(trajectory)
    if any(len(group) != 4 for group in groups.values()):
        raise ValueError("Require exactly four trajectories per artwork")
    if rewards:
        import math

        for group in groups.values():
            if not all(isinstance(t["reward"], (int, float)) and math.isfinite(t["reward"]) for t in group):
                raise ValueError("Reward must be finite")
            baseline = sum(t["reward"] for t in group) / 4
            for t in group:
                t["advantage"] = t["reward"] - baseline
    for t in trajectories:
        t["loss_coefficient_per_token"] = 1 / (len(trajectories) * t["completion_tokens"])
    return {
        "objective": "mean_trajectory(sum_turn_token_clipped_PPO / total_trajectory_completion_tokens)",
        "advantages": "terminal reward minus four-trajectory artwork mean; no std normalization",
        "reward_file": str(rewards) if rewards else None,
        "reward_sha256": hashlib.sha256(rewards.read_bytes()).hexdigest() if rewards else None,
        "trajectories": trajectories,
        "sequence_lengths": sorted(t["sequence_tokens"] for e in trajectories for t in e["turns"]),
        "optimizer_steps": 1,
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": 0},
        "lora": {"rank": 8, "alpha": 16, "target": "o_proj"},
        "activation_checkpointing": "full, every block",
        "dtype": "existing production BF16 forward/FP32 masters unchanged",
    }


def execute(plan, output):
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch._dynamo.utils import counters

    from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, LoRAConfig, ModelConfig
    from prime_rl.trainer.lora import get_lora_state, save_lora_config
    from prime_rl.trainer.model import setup_model
    from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
    from prime_rl.trainer.parallel_dims import get_parallel_dims
    from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.utils.logger import setup_logger
    from tools.painting_dataset.agentic_batch import turn_to_microbatch

    started = time.monotonic()
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
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in params):
        raise ValueError("Expected existing FP32 LoRA master parameters")
    optimizer = torch.optim.AdamW(params, lr=plan.get("optimizer", {}).get("learning_rate", 1e-5), weight_decay=0)
    rows = []
    samples = [
        (trajectory, turn)
        for trajectory in plan["trajectories"]
        if trajectory["advantage"] != 0
        for turn in trajectory["turns"]
    ]
    if not samples:
        raise ValueError("No nonzero advantage, refusing empty update")
    # Longest context first exposes capacity before spending time on smaller samples.
    samples.sort(key=lambda pair: pair[1]["sequence_tokens"], reverse=True)

    def replay_loss(trajectory, turn):
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
        old = tensors["inference_logprobs"].roll(-1, dims=-1).cuda()
        logratio = result["logprobs"][mask] - old[mask]
        ratio = logratio.exp()
        advantage = trajectory["advantage"]
        loss = (
            -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).sum()
            * trajectory["loss_coefficient_per_token"]
        )
        if not torch.isfinite(loss) or not torch.isfinite(ratio).all():
            raise ValueError("Nonfinite replay objective")
        return loss, ratio, logratio

    for trajectory, turn in samples:
        loss, ratio, logratio = replay_loss(trajectory, turn)
        directory = Path(turn["directory"])
        advantage = trajectory["advantage"]
        loss.backward()
        rows.append(
            {
                "turn": str(directory),
                "loss": loss.item(),
                "advantage": advantage,
                "ratio_min": ratio.min().item(),
                "ratio_max": ratio.max().item(),
                "sampled_k3": (ratio - 1 - logratio).mean().item(),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        if dist.get_rank() == 0:
            (output / "progress.json").write_text(json.dumps(rows, indent=2))
            print(rows[-1], flush=True)
        del loss, ratio, logratio
    norm = sum(
        (p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).float().square().sum()
        for p in params
        if p.grad is not None
    )
    dist.all_reduce(norm)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid accumulated gradient norm")
    optimizer.step()
    state = get_lora_state().adapter_state_dict()
    full = {
        k: (v.full_tensor() if hasattr(v, "full_tensor") else v).detach().cpu().contiguous().clone()
        for k, v in state.items()
    }
    if not all(torch.isfinite(v).all() for v in full.values()):
        raise ValueError("Nonfinite adapter")
    if dist.get_rank() == 0:
        destination = output / plan.get("adapter_name", "adapter-painting-step1")
        destination.mkdir()
        save_file(full, str(destination / "adapter_model.safetensors"))
        save_lora_config(model, destination, rank=8, alpha=16, dropout=0)
        (output / "complete.json").write_text(
            json.dumps({"optimizer_steps": 1, "gradient_norm": norm.sqrt().item(), "samples": len(samples)}, indent=2)
        )
    # Adapter is already exported before this read-only acceptance pass. Never repeat the optimizer if evaluation fails.
    optimizer.zero_grad(set_to_none=True)
    after_rows = []
    with torch.no_grad():
        for trajectory, turn in samples:
            loss, ratio, logratio = replay_loss(trajectory, turn)
            after_rows.append(
                {
                    "turn": turn["directory"],
                    "loss": loss.item(),
                    "ratio_min": ratio.min().item(),
                    "ratio_max": ratio.max().item(),
                    "sampled_k3": (ratio - 1 - logratio).mean().item(),
                }
            )
            if dist.get_rank() == 0:
                (output / "after-objective-progress.json").write_text(json.dumps(after_rows, indent=2))
            del loss, ratio, logratio
    if dist.get_rank() == 0:
        before = -sum(row["loss"] for row in rows)
        after = -sum(row["loss"] for row in after_rows)
        (output / "objective-comparison.json").write_text(
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "delta": after - before,
                    "wall_seconds_including_startup": time.monotonic() - started,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    "gradient_norm": norm.sqrt().item(),
                },
                indent=2,
            )
        )
    (output / f"compiler-rank{dist.get_rank()}.json").write_text(
        json.dumps({str(k): dict(v) for k, v in counters.items()}, indent=2)
    )
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rewards", type=Path, help="Explicit approved terminal reward map keyed by absolute episode path"
    )
    parser.add_argument("--execute", action="store_true", help="Execute one optimizer step; only after reward approval")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adapter-name", default="adapter-painting-step1")
    args = parser.parse_args()
    plan = build_plan(args.pilot, args.rewards, args.learning_rate)
    if Path(args.adapter_name).name != args.adapter_name or args.adapter_name in (".", ".."):
        parser.error("--adapter-name must be a simple directory name")
    plan["adapter_name"] = args.adapter_name
    if args.execute and not args.rewards:
        parser.error("--execute requires approved --rewards")
    args.output.mkdir(exist_ok=True)
    if (args.output / "complete.json").exists():
        raise ValueError("Refusing duplicate completed update")
    if not args.execute or __import__("os").environ.get("RANK", "0") == "0":
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    if args.execute:
        execute(plan, args.output)
    else:
        print(
            json.dumps(
                {
                    "trajectories": len(plan["trajectories"]),
                    "turns": len(plan["sequence_lengths"]),
                    "sequence_min_max": [min(plan["sequence_lengths"]), max(plan["sequence_lengths"])],
                    "optimizer_started": False,
                }
            )
        )


if __name__ == "__main__":
    main()
