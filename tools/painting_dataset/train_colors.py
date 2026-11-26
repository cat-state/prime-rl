"""Scored-image LoRA updates with optional group-centered gradient accumulation."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from prime_rl.configs.trainer import CompileConfig, LoRAConfig, ModelConfig
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
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--capture-gradients", action="store_true", help="Preserve pre-optimizer gradient shards for numerical diagnosis")
    parser.add_argument("--aot-eager", action="store_true", help="Diagnostic graph capture without Inductor code generation")
    parser.add_argument("--dynamo-eager", action="store_true", help="Diagnostic Dynamo capture without AOTAutograd")
    parser.add_argument("--forward-only", action="store_true", help="Diagnostic fixed-token forward comparison without backward")
    parser.add_argument("--attention-boundary", action="store_true", help="Diagnostic graph break around independently compiled attention")
    parser.add_argument("--compiler-audit", action="store_true", help="Save Dynamo counters in a separate diagnostic run")
    parser.add_argument("--capture-layer-outputs", action="store_true", help="Diagnostic sampled activation capture outside compiled layer calls")
    args = parser.parse_args()
    if args.aot_eager and not (args.compile and args.capture_gradients):
        parser.error("--aot-eager requires --compile --capture-gradients")
    if args.dynamo_eager and (args.aot_eager or not (args.compile and args.capture_gradients)):
        parser.error("--dynamo-eager requires --compile --capture-gradients and excludes --aot-eager")
    setup_logger("info")
    setup_torch_distributed()
    torch.manual_seed(42)
    root = args.dataset
    destination = root / ("adapter-group-step1" if args.group_centered else "adapter-step4")
    if args.compile:
        destination = destination.with_name(destination.name + "-compiled")
    metrics_path = root / ("scored-update-metrics-compiled.json" if args.compile else "scored-update-metrics.json")
    if args.capture_gradients:
        destination = destination.with_name(destination.name + "-diagnostic")
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-diagnostic")
    if args.aot_eager:
        destination = destination.with_name(destination.name + "-aot-eager")
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-aot-eager")
    if args.dynamo_eager:
        destination = destination.with_name(destination.name + "-dynamo-eager")
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-dynamo-eager")
    if args.forward_only:
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-no-grad-forward")
    if args.attention_boundary:
        if not (args.compile and args.capture_gradients):
            parser.error("--attention-boundary requires --compile --capture-gradients")
        destination = destination.with_name(destination.name + "-attention-boundary")
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-attention-boundary")
    if args.compiler_audit:
        destination = destination.with_name(destination.name + "-audit")
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-audit")
    layer_capture = None
    if args.capture_layer_outputs:
        if not (args.forward_only and args.capture_gradients and args.compiler_audit):
            parser.error("--capture-layer-outputs requires --forward-only --capture-gradients --compiler-audit")
        from tools.capture_inkling_execution_layers import install_capture
        destination = destination.with_name(destination.name + "-layers")
        metrics_path = metrics_path.with_stem(metrics_path.stem + "-layers")
        layer_capture = install_capture()
    if (destination.exists() and not args.forward_only) or metrics_path.exists():
        raise ValueError("Refusing to overwrite prior adapter")
    config = ModelConfig(name="/dev/shm/inkling2-loop-bf16", conversion_dir=Path("/dev/shm/inkling2-training"),
                         compile=CompileConfig(fullgraph=False) if args.compile else None, ep=1, cp=1, offload="none",
                         lora=LoRAConfig(rank=8, alpha=16, target_modules=["o_proj"]))
    if args.attention_boundary:
        from prime_rl.trainer.models.inkling2 import modeling_inkling2

        original_attention = modeling_inkling2.flex_attention
        @torch.compiler.disable(recursive=False)
        def attention_boundary(*positional, **kwargs):
            return original_attention(*positional, **kwargs)
        modeling_inkling2.flex_attention = attention_boundary
    original_compile = torch.nn.Module.compile
    def diagnostic_compile(module, *positional, **kwargs):
        kwargs.pop("mode", None)
        return original_compile(module, *positional, backend="eager" if args.dynamo_eager else "aot_eager", **kwargs)
    with patch.object(torch.nn.Module, "compile", diagnostic_compile) if args.aot_eager or args.dynamo_eager else nullcontext():
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
    if args.capture_layer_outputs:
        samples = samples[:1]
    if args.forward_only:
        torch.set_grad_enabled(False)
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
        if args.forward_only:
            metrics.append({"id": row["id"], "loss": loss.item(),
                            "sampled_logprobs": output["logprobs"][mask].detach().float().cpu().tolist()})
            if dist.get_rank() == 0:
                metrics_path.write_text(json.dumps(metrics, indent=2))
            del output, loss, ratio
            continue
        loss.backward()
        def local(t):
            return t.to_local() if hasattr(t, "to_local") else t
        norm = sum(local(p.grad).float().square().sum() for p in params if p.grad is not None)
        dist.all_reduce(norm)
        if not torch.isfinite(norm) or (score > 0 and norm <= 0):
            raise ValueError("Invalid scored image gradients")
        if not args.group_centered or index == len(samples) - 1:
            if args.capture_gradients:
                gradients = {name: local(p.grad).detach().cpu().contiguous().clone()
                             for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
                save_file(gradients, str(root / f"{destination.name}-sample{index}-rank{dist.get_rank()}-grads.safetensors"))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics.append({"id": row["id"], "reward": score, "advantage": advantage,
                        "loss": loss.item(), "gradient_norm": norm.sqrt().item()})
        if args.capture_gradients:
            metrics[-1]["sampled_logprobs"] = output["logprobs"][mask].detach().float().cpu().tolist()
        if dist.get_rank() == 0:
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print({k: v for k, v in metrics[-1].items() if k != "sampled_logprobs"}, flush=True)
    if args.compiler_audit:
        from torch._dynamo.utils import counters

        audit = {str(group): {str(key): value for key, value in counts.items()} for group, counts in counters.items()}
        (root / f"{destination.name}-rank{dist.get_rank()}-compiler.json").write_text(json.dumps(audit, indent=2))
    if layer_capture is not None and dist.get_rank() == 0:
        save_file(layer_capture, str(root / f"{destination.name}-activations.safetensors"))
    if args.forward_only:
        dist.destroy_process_group()
        return
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
