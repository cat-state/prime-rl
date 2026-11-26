"""Small, isolated attention compilation diagnostic; no production changes."""

import argparse
import json
from pathlib import Path

import torch

from prime_rl.trainer.models.inkling2 import modeling_inkling2 as modeling
from prime_rl.trainer.models.inkling2.configuration_inkling2 import Inkling2TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evidence")
    torch.manual_seed(42)
    config = Inkling2TextConfig(hidden_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=64,
        local_layer_ids=[], rel_extent=128, sliding_window_size=128)
    module = modeling.Inkling2Attention(config, 0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        module.rel_logits_proj.normal_(std=0.02)
    x = torch.randn(1, 256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    position = torch.arange(256, device="cuda").unsqueeze(0)
    docs = torch.zeros_like(position)
    cu = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    mask = modeling._build_block_masks(docs, 128)["full"]
    original = modeling.flex_attention
    results = {}
    tensors = {}
    for mode in ("eager", "outer_eager", "outer_inductor", "nested_inductor"):
        module.zero_grad(set_to_none=True)
        x.grad = None
        if mode == "nested_inductor":
            modeling.flex_attention = torch.compiler.nested_compile_region(modeling._flex_attention)
        else:
            modeling.flex_attention = original
        fn = module if mode == "eager" else torch.compile(
            module, backend="eager" if mode == "outer_eager" else "inductor", fullgraph=True)
        try:
            output = fn(x, x, x, position, cu, docs, mask)
            output.float().square().mean().backward()
            tensors[mode] = (output.detach().float(), x.grad.detach().float().clone())
            ref = tensors["eager"]
            results[mode] = {}
            for name, actual, expected in zip(("output", "input_gradient"), tensors[mode], ref, strict=True):
                results[mode][name] = {
                    "max_abs": (actual - expected).abs().max().item(),
                    "relative_l2": ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item(),
                    "finite": bool(torch.isfinite(actual).all()),
                }
        except Exception as exc:
            results[mode] = {"error": repr(exc)}
        print(mode, results[mode], flush=True)
    # Isolate the kernel from the surrounding projections, norms, casts and gates.
    q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    rel = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16) * 0.1

    def kernel(q, k, v, rel):
        out, aux = original(q, k, v, score_mod=modeling._relative_bias_score_mod(rel, 128),
                            block_mask=mask, scale=0.125, enable_gqa=True,
                            return_aux=modeling.AuxRequest(lse=True))
        return out, aux.lse

    reference = kernel(q, k, v, rel)
    for backend in ("eager", "inductor"):
        actual = torch.compile(kernel, backend=backend, fullgraph=True)(q, k, v, rel)
        result = {name: {"max_abs": (a.float()-b.float()).abs().max().item(),
                        "relative_l2": ((a.float()-b.float()).norm()/b.float().norm()).item()}
                  for name, a, b in zip(("output", "lse"), actual, reference, strict=True)}
        results[f"kernel_only_{backend}"] = result
        print(f"kernel_only_{backend}", result, flush=True)
    results["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
