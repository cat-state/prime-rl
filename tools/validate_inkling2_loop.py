"""Compare shared-loop gradients through PrimeRL FSDP and activation checkpointing.

Run: uv run torchrun --standalone --nproc-per-node=2 tools/validate_inkling2_loop.py OUTPUT_DIR
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from prime_rl.configs.trainer import CompileConfig, ModelConfig
from prime_rl.trainer.model import apply_compile, setup_fsdp
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopTextConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--fp32-reference", action="store_true", help="Use replicated FP32 masters and FSDP BF16 compute"
    )
    parser.add_argument("--checkpoint-reference", action="store_true")
    parser.add_argument("--router-replay", action="store_true")
    parser.add_argument("--preprocessed-embeds", action="store_true")
    args = parser.parse_args()
    setup_logger("info")
    setup_torch_distributed()
    torch.manual_seed(51)
    config = Inkling2LoopTextConfig(
        vocab_size=128,
        eos_token_id=127,
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=32,
        d_rel=4,
        rel_extent=16,
        sliding_window_size=16,
        local_layer_ids=[0, 2, 4, 6],
        n_layers_per_attn_res_block=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        loop_layer_index_start=2,
        loop_layer_index_end=6,
        loop_gate_rank=8,
    )
    reference = Inkling2LoopForCausalLM.from_config(config, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "message" in name:
                parameter.normal_(std=0.1)
        # Give the selected experts a margin so this sharding test does not
        # conflate floating-point fusion with discrete top-k route changes.
        if not args.router_replay:
            for layer in reference.model.layers:
                layer.mlp.router.selection_bias.copy_(torch.arange(4, device="cuda") * 2.0)
    inject_prime_lm_head(reference)
    # Match the trainer's FP32 master weights and BF16 FSDP compute policy.
    sharded = copy.deepcopy(reference).float()
    if args.fp32_reference:
        reference.float()
    ids = torch.randint(0, 128, (1, 16), device="cuda")
    positions = torch.arange(8, device="cuda").repeat(2).unsqueeze(0)
    lengths = torch.tensor([8, 8], device="cuda")
    routes = torch.rand(1, 16, 16, 4, device="cuda").argsort(-1)[..., :2].contiguous() if args.router_replay else None

    prepared_embeds = None
    if args.preprocessed_embeds:
        with torch.no_grad():
            prepared_embeds = reference.model.embed_norm(reference.model.embed_tokens(ids)).detach()
            # Simulate frozen image features inserted after text normalization.
            prepared_embeds[:, 2:6] = torch.randn_like(prepared_embeds[:, 2:6]) * 3

    def loss(model):
        inputs = {"input_ids": ids}
        if prepared_embeds is not None:
            inputs = {"inputs_embeds": prepared_embeds, "inputs_embeds_preprocessed": True}
        output = model(**inputs, position_ids=positions, seq_lens=lengths, labels=ids, routed_experts=routes)
        return output["loss"] / ids.numel()

    if args.checkpoint_reference:
        for i, layer in enumerate(reference.model.layers):
            reference.model.layers[i] = checkpoint_wrapper(layer)
    if args.compile:
        apply_compile(reference, CompileConfig(fullgraph=True))
    if args.fp32_reference:
        reference_config = ModelConfig(name="tiny-inkling2-loop", ep=1, dp_replicate=dist.get_world_size())
        setup_fsdp(reference, reference_config, get_parallel_dims(reference_config))
    expected_loss = loss(reference)
    expected_loss.backward()
    trainer_config = ModelConfig(name="tiny-inkling2-loop", ep=1)
    parallel_dims = get_parallel_dims(trainer_config)
    for i, layer in enumerate(sharded.model.layers):
        sharded.model.layers[i] = checkpoint_wrapper(layer)
    if args.compile:
        apply_compile(sharded, CompileConfig(fullgraph=True))
    setup_fsdp(sharded, trainer_config, parallel_dims)
    actual_loss = loss(sharded)
    actual_loss.backward()
    reference_params = {
        name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): p
        for name, p in reference.named_parameters()
    }

    def full_gradient(parameter):
        gradient = parameter.grad
        return gradient.full_tensor().float() if hasattr(gradient, "full_tensor") else gradient.float()

    total_norm = torch.sqrt(sum(full_gradient(p).square().sum() for p in reference.parameters() if p.grad is not None))
    errors = {}
    for name, parameter in sharded.named_parameters():
        name = name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        expected = reference_params[name].grad
        if expected is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        expected = full_gradient(reference_params[name])
        actual = parameter.grad.full_tensor().float()
        denominator = max(expected.float().norm().item(), total_norm.item() * 0.001)
        errors[name] = (actual - expected.float()).norm().item() / denominator
    relative_loss = abs(actual_loss.item() - expected_loss.item()) / abs(expected_loss.item())
    metrics = {
        "reference_loss": expected_loss.item(),
        "fsdp_loss": actual_loss.item(),
        "relative_loss_error": relative_loss,
        "worst_gradient_error": max(errors.values()),
        "compile": args.compile,
        "fp32_reference": args.fp32_reference,
        "checkpoint_reference": args.checkpoint_reference,
        "router_replay": args.router_replay,
        "preprocessed_embeds": args.preprocessed_embeds,
        "gradient_errors": errors,
    }
    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(json.dumps(metrics, indent=2))
    # Compare the same arithmetic with/without sharding. Eager versus compiled
    # arithmetic can change top-k expert choices and is a separate comparison.
    assert relative_loss < 1e-5, metrics
    assert max(errors.values()) < 0.02, metrics
    optimizer = torch.optim.SGD(sharded.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(loss(sharded)), "non-finite loss after optimizer step"
    if dist.get_rank() == 0:
        print(json.dumps({k: v for k, v in metrics.items() if k != "gradient_errors"}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
