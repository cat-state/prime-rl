"""BF16 loop inference with independent KV/short-convolution caches per visit.

Inference unrolls the core visits; checkpoint loading fans each shared trainer
weight out to its visits. This trades inference memory for the existing vLLM
cache and weight-reload machinery.
"""

import copy
import re

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase

from .configs import Inkling2MMConfig
from .model import Inkling2AttnRes, Inkling2DecoderLayer, Inkling2ForCausalLM, Inkling2Model
from .ops.attention import compute_log_scaling_tau


class Inkling2LoopMMConfig(Inkling2MMConfig):
    model_type = "inkling2_loop"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Inkling2LoopArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self):
        config = self.hf_text_config
        loop = config.loop_config
        return config.num_hidden_layers + (loop["core_end"] - loop["core_start"]) * (loop["num_passes"] - 1)


class MessageController(nn.Module):
    def __init__(self, hidden, rank):
        super().__init__()
        self.down_weight = nn.Parameter(torch.empty(rank, 2 * hidden), requires_grad=False)
        self.gain_weight = nn.Parameter(torch.empty(hidden, rank, dtype=torch.float32), requires_grad=False)
        self.forget_weight = nn.Parameter(torch.empty(hidden, rank, dtype=torch.float32), requires_grad=False)

    def forward(self, dest, message):
        pair = torch.stack((dest.float(), message.float()), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        hidden = F.linear(pair.flatten(1).to(self.down_weight.dtype), self.down_weight).float()
        gain = F.linear(hidden, self.gain_weight).tanh()
        forget = F.linear(hidden, self.forget_weight).tanh()
        return (1 - forget) * dest + gain * message


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__(config, layer_id, quant_config, prefix)
        self.mlp.gate.weight.data = self.mlp.gate.weight.data.float()
        for conv in (self.attn_sconv, self.mlp_sconv, self.attn.k_sconv, self.attn.v_sconv):
            if not conv.dynamic:
                conv.weight.data = conv.weight.data.float()
        loop = config.loop_config
        original = config.original_layer_ids[layer_id]
        if loop["core_start"] <= original < loop["core_end"]:
            self.controllers = nn.ModuleDict(
                {
                    stream: MessageController(config.hidden_size, loop["gate_rank"])
                    for stream in ("attn", "k", "v", "mlp")
                }
            )


class Inkling2LoopModel(Inkling2Model):
    decoder_layer_class = Inkling2LoopDecoderLayer

    def __init__(self, config, quant_config, prefix):
        loop = config.loop_config
        start, end, passes = loop["core_start"], loop["core_end"], loop["num_passes"]
        if not 0 < start < end <= config.num_hidden_layers or passes < 1:
            raise ValueError("Invalid Inkling2 loop bounds or pass count")
        if start % config.n_layers_per_attn_res_block or end % config.n_layers_per_attn_res_block:
            raise ValueError("Loop bounds must align with attention-residual blocks")
        self.visits = list(range(start)) + list(range(start, end)) * passes + list(range(end, config.num_hidden_layers))
        expanded = copy.deepcopy(config)
        expanded.original_layer_ids = self.visits
        expanded.num_hidden_layers = len(self.visits)
        expanded.local_layer_ids = [i for i, original in enumerate(self.visits) if original in config.local_layer_ids]
        super().__init__(expanded, quant_config, prefix)
        self.config = config
        self.attn_res = Inkling2AttnRes(config)
        shape = (2 * (end - start), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(torch.zeros(shape, dtype=torch.float32), False)
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(torch.zeros(shape, dtype=torch.float32), False)

    def mix(self, stack, last, queries):
        # Queries here are already scaled by hidden_size**-0.5.
        values = torch.stack([*stack, last], dim=-2).float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        scores = torch.einsum("tnd,qd->tqn", normalized, queries)
        return torch.einsum("tqn,tnd->qtd", scores.softmax(-1), normalized if self.normalize_values else values)

    def forward(self, positions, inputs_embeds):
        if self.aux_hidden_state_layers:
            raise ValueError("Loop inference does not yet support speculative hidden-state capture")
        loop = self.config.loop_config
        start, end = loop["core_start"], loop["core_end"]
        hidden, blocks = inputs_embeds, []
        tau = None
        if self.config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha)
        for visit, original in enumerate(self.visits):
            layer = self.layers[visit]
            messages = None
            if original == start:
                if visit == start:
                    pre_hidden, pre_blocks = hidden, list(blocks)
                else:
                    previous_hidden, previous_blocks = hidden, blocks[len(pre_blocks) + 1 :]
                    hidden, blocks = pre_hidden, list(pre_blocks)
            if start <= original < end and visit >= end:
                offset = 2 * (original - start)
                queries = (
                    torch.stack(
                        (
                            self.middle_cycle_cross_pass_queries[offset],
                            self.middle_cycle_cross_pass_kv_queries[offset],
                            self.middle_cycle_cross_pass_kv_queries[offset + 1],
                            self.middle_cycle_cross_pass_queries[offset + 1],
                        )
                    )
                    * self.config.hidden_size**-0.5
                )
                messages = self.mix(previous_blocks, previous_hidden, queries)
            if original == 0:
                inputs = [hidden.float()] * 3
            else:
                inputs = self.mix(blocks, hidden, self.attn_res.attn_queries[original - 1])
                if layer.is_local:
                    inputs = [inputs[0]] * 3
            if messages is not None:
                inputs = [
                    layer.controllers[name](x, message)
                    for name, x, message in zip(("attn", "k", "v"), inputs, messages)
                ]
            inputs = [layer.attn_norm(x, 0 if layer.is_local else i) for i, x in enumerate(inputs)]
            attn = layer.attn(positions, *inputs, tau=tau)
            attn = self._hidden_conv(layer.attn_sconv, tensor_model_parallel_all_reduce(attn), positions)
            if original % self.block_size == 0:
                blocks.append(hidden)
                residual = attn.float()
            else:
                residual = hidden.float() + attn.float()
            mlp_input = self.mix(blocks, residual, self.attn_res.mlp_queries[original])[0]
            if messages is not None:
                mlp_input = layer.controllers["mlp"](mlp_input, messages[3])
            mlp_input = layer.mlp_norm(mlp_input)
            logits = F.linear(mlp_input.float(), layer.mlp.gate.weight)
            mlp_out = layer.mlp.experts(hidden_states=mlp_input, router_logits=logits)
            mlp_out = self._hidden_conv(layer.mlp_sconv, mlp_out, positions)
            hidden = (residual + mlp_out.float()).to(inputs_embeds.dtype)
        return self.norm(self.mix(blocks, hidden, self.attn_res.output_query)[0])


class Inkling2LoopForCausalLM(Inkling2ForCausalLM, SupportsLoRA):
    # External embeddings must already include text embed_norm and replacement
    # image features, matching the reference multimodal model's LM boundary.
    accepts_preprocessed_prompt_embeds = True
    model_class = Inkling2LoopModel

    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.lora_config is not None and vllm_config.lora_config.target_modules != ["wo_ud"]:
            raise ValueError("Inkling loop LoRA requires lora_target_modules=['wo_ud'] (output projections only)")
        if vllm_config.quant_config is not None:
            raise ValueError("Loop training inference requires a dequantized BF16 checkpoint")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.lora_config is not None:
            for layer in self.model.layers:
                layer.attn.output_lora = True

    def load_weights(self, weights):
        for layer in self.model.layers:
            if len(layer.attn.split_sizes) == 7:
                parameter = layer.attn.qkvr.weight
                padding = torch.zeros(
                    layer.attn.split_sizes[-1] * self.model.tp_size,
                    parameter.shape[1],
                    dtype=parameter.dtype,
                    device="cuda",
                )
                parameter.weight_loader(parameter, padding, 6)

        def expanded_weights():
            for name, weight in weights:
                match = re.search(r"\.layers\.(\d+)\.", name)
                if match is None:
                    yield name, weight
                else:
                    for visit, original in enumerate(self.model.visits):
                        if original == int(match[1]):
                            yield name[: match.start(1)] + str(visit) + name[match.end(1) :], weight

        original_config = self.text_config
        self.text_config = copy.deepcopy(original_config)
        self.text_config.num_hidden_layers = len(self.model.visits)
        try:
            return super().load_weights(expanded_weights())
        finally:
            self.text_config = original_config
