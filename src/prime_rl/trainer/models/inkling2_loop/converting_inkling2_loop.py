"""Load released NVFP4/MXFP8 experts; export dequantized HF weights for training."""

from dataclasses import dataclass

import torch

from prime_rl.trainer.models.conversion_ops import ConvOp, MapValue, Rename
from prime_rl.trainer.models.inkling2.converting_inkling2 import conversion_chain as base_conversion_chain


@dataclass
class DequantizeExperts(ConvOp):
    name: str

    def hf_to_prime(self, sd):
        scale = sd.pop(self.name + "_scale", None)
        if scale is None:
            return
        weight = sd[self.name]
        amax = sd.pop(self.name + "_amax", None)
        denominator = sd.pop(self.name + "_outer_scale_denominator", None)
        experts, rows, columns = weight.shape
        if amax is not None:
            if denominator is None:
                raise ValueError(f"Missing NVFP4 outer scale denominator for {self.name}")
            columns *= 2
            values = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=weight.device
            )
            byte = torch.arange(256, device=weight.device)
            lookup = torch.stack([values[byte & 15], values[byte >> 4]], dim=-1)
            factor = amax.float() / denominator.float()
        output = torch.empty(experts, rows, columns, dtype=torch.bfloat16, device=weight.device)
        for start in range(0, experts, 8):
            chunk = slice(start, start + 8)
            scales = scale[chunk].float()
            if amax is None:
                decoded = weight[chunk].float().view(*scales.shape, 32) * torch.exp2(scales - 127)[..., None]
            else:
                decoded = lookup[weight[chunk].to(torch.uint8).long()].view(*scales.shape, 16)
                decoded = decoded * scales[..., None] * factor[chunk, None, None, None]
            output[chunk] = decoded.reshape(-1, rows, columns).to(output.dtype)
        sd[self.name] = output

    def prime_to_hf(self, sd):
        # Export BF16 tensors rather than re-quantizing updated training weights.
        pass


class MessageQueries(ConvOp):
    hf = "model.middle_cycle_cross_pass_queries"
    hf_kv = "model.middle_cycle_cross_pass_kv_queries"
    prime = "model.message_input.query"

    def hf_to_prime(self, sd):
        if self.hf not in sd:
            return
        queries = sd.pop(self.hf).unflatten(0, (-1, 2))
        kv = sd.pop(self.hf_kv).unflatten(0, (-1, 2))
        sd[self.prime] = torch.cat([queries[:, :1], kv, queries[:, 1:]], dim=1).float()

    def prime_to_hf(self, sd):
        if self.prime not in sd:
            return
        queries = sd.pop(self.prime).float()
        sd[self.hf] = queries[:, [0, 3]].flatten(0, 1).contiguous()
        sd[self.hf_kv] = queries[:, 1:3].flatten(0, 1).contiguous()


def conversion_chain(config):
    ops = [
        DequantizeExperts(f"model.llm.layers.{i}.mlp.experts.{weight}")
        for i in range(config.num_hidden_layers)
        for weight in ("w13_weight", "w2_weight")
    ]
    ops += [op for op in base_conversion_chain(config) if not isinstance(op, MapValue)]
    ops.append(MessageQueries())
    for i in range(config.num_hidden_layers):
        names = [
            "mlp.router.gate.weight",
            "post_attention_convolution.conv1d.weight",
            "post_mlp_convolution.conv1d.weight",
        ]
        if i in config.local_layer_ids:
            names += ["self_attn.k_sconv.conv1d.weight", "self_attn.v_sconv.conv1d.weight"]
        for name in names:
            ops.append(
                MapValue(prime=f"model.layers.{i}.{name}", forward=lambda t: t.float(), backward=lambda t: t.float())
            )
    for i in range(config.loop_layer_index_start, config.loop_layer_index_end):
        prefix = f"model.layers.{i}"
        for hf, prime in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            for source, target in (
                ("down_weight", "down_proj"),
                ("gain_weight", "message_gate_proj"),
                ("forget_weight", "forget_gate_proj"),
            ):
                name = f"{prefix}.{prime}_message_gate.{target}.weight"
                ops.append(Rename(f"{prefix}.controllers.{hf}.{source}", name))
                if target != "down_proj":
                    ops.append(MapValue(prime=name, forward=lambda t: t.float(), backward=lambda t: t.float()))
    return ops
