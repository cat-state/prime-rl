"""Replay a real first layer and compare BF16 paths to a diagnostic FP32 oracle."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
from prime_rl.trainer.models.inkling2 import modeling_inkling2 as modeling
from prime_rl.trainer.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig
from prime_rl.trainer.models.inkling2_loop.converting_inkling2_loop import conversion_chain
from prime_rl.trainer.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopDecoderLayer


class FP32Experts:
    """Same expert equations/routes, with explicit FP32 per-expert GEMMs."""
    token_group_alignment = 8

    def __call__(self, experts, x, counts):
        outputs = []
        start = 0
        for expert, count in enumerate(counts.cpu().tolist()):
            values = x[start:start + count]
            start += count
            if count:
                gate = F.linear(values, experts.gate_proj[expert])
                up = F.linear(values, experts.up_proj[expert])
                outputs.append(F.linear(experts.activation.apply(gate, up), experts.down_proj[expert]))
        return torch.cat(outputs, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evidence")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Path("/dev/shm/inkling2-loop-bf16")
    config = Inkling2LoopConfig(**json.loads((source / "config.json").read_text()))
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for key, filename in index.items():
        if key.startswith("model.llm.layers.0."):
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                state[key] = handle.get_tensor(key)
    apply_hf_to_prime(state, conversion_chain(config))
    state = {key.removeprefix("model.layers.0."): value for key, value in state.items()}
    with torch.device("meta"):
        layer = Inkling2LoopDecoderLayer(config, 0)
    layer.load_state_dict(state, strict=True, assign=True)
    layer.mlp.tokens_per_expert = torch.zeros(config.n_routed_experts)
    layer.mlp.routing_confidence_sum = torch.tensor(0.)
    layer = layer.cuda().eval()
    del state
    inputs = {k: v.cuda() for k, v in load_file(str(args.capture)).items()}
    masks = modeling._build_block_masks(inputs["doc_ids"], config.sliding_window_size)

    def run(function, fp32=False):
        return function(inputs["hidden"].float() if fp32 else inputs["hidden"],
                        inputs["blocks"].float() if fp32 else inputs["blocks"],
                        None, inputs["mlp_query"], inputs["position_ids"], inputs["cu_seqlens"],
                        inputs["doc_ids"], masks, routed_experts=inputs["routes"])[0]

    outputs = {}
    with torch.no_grad():
        outputs["bf16_eager"] = run(layer).float().cpu()
        outputs["bf16_compiled"] = run(torch.compile(layer, fullgraph=True)).float().cpu()
        layer.float()
        layer.mlp.experts.compute = FP32Experts()
        # Eager dense attention avoids TF32 Triton selection for this small oracle.
        # All routing indices and original checkpoint values remain identical.
        modeling.flex_attention = modeling._flex_attention
        outputs["fp32_oracle"] = run(layer, fp32=True).float().cpu()
    outputs["captured_eager"] = inputs["output_hidden"].float().cpu()
    comparisons = {}
    for name, reference in (("bf16_eager", "captured_eager"), ("bf16_compiled", "bf16_eager"),
                            ("bf16_eager", "fp32_oracle"), ("bf16_compiled", "fp32_oracle")):
        a, b = outputs[name].double(), outputs[reference].double()
        comparisons[f"{name}_vs_{reference}"] = {
            "relative_l2": ((a-b).norm()/b.norm()).item(),
            "max_abs": (a-b).abs().max().item(),
            "reference_rms": b.square().mean().sqrt().item(),
            "finite": bool(torch.isfinite(a).all()),
        }
    args.output.write_text(json.dumps(comparisons, indent=2))
    save_file(outputs, str(args.output.with_suffix(".safetensors")))
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
