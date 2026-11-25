"""Inkling 2 Mini's shared-weight loop and cross-pass messages for text training."""

import re

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.inkling2.modeling_inkling2 import (
    Inkling2BlockAttentionResidual,
    Inkling2DecoderLayer,
    Inkling2ForCausalLM,
    Inkling2Model,
    Inkling2UnweightedRMSNorm,
)
from prime_rl.trainer.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from prime_rl.trainer.models.inkling2_loop.converting_inkling2_loop import conversion_chain


class MessageGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = Inkling2UnweightedRMSNorm(config.rms_norm_eps)
        self.down_proj = nn.Linear(2 * config.hidden_size, config.loop_gate_rank, bias=False)
        self.forget_gate_proj = nn.Linear(config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32)
        self.message_gate_proj = nn.Linear(config.loop_gate_rank, config.hidden_size, bias=False, dtype=torch.float32)

    def forward(self, hidden_states, message):
        hidden_states, message = hidden_states.float(), message.float()
        controller_input = torch.cat([self.norm(hidden_states), self.norm(message)], dim=-1)
        controller = self.down_proj(controller_input.to(self.down_proj.weight.dtype)).float()
        forget = F.linear(controller, self.forget_gate_proj.weight.float()).tanh()
        gain = F.linear(controller, self.message_gate_proj.weight.float()).tanh()
        return (1 - forget) * hidden_states + gain * message


class MessageInput(Inkling2BlockAttentionResidual):
    def __init__(self, config):
        super().__init__(config.hidden_size, 1e-6)
        count = config.loop_layer_index_end - config.loop_layer_index_start
        self.query = nn.Parameter(torch.zeros(count, 4, config.hidden_size, dtype=torch.float32))


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    preserve_input_dtypes = True

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp.router.gate_pre_unsharded = True
        self.mlp.router.gate.float()
        self.post_attention_convolution.float()
        self.post_mlp_convolution.float()
        if self.is_sliding:
            self.self_attn.k_sconv.float()
            self.self_attn.v_sconv.float()
        if config.loop_layer_index_start <= layer_idx < config.loop_layer_index_end:
            for name in ("q", "k", "v", "mlp"):
                setattr(self, f"{name}_message_gate", MessageGate(config))

    def forward(
        self,
        hidden_states,
        block_states,
        attn_query,
        mlp_query,
        position_ids,
        cu_seqlens,
        doc_ids,
        block_masks,
        routed_experts=None,
        messages=None,
        **kwargs,
    ):
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is None:
            q_hidden = k_hidden = v_hidden = hidden_states
        else:
            inputs = self.attention_input(attn_query, block_states, hidden_states)
            if self.is_sliding:
                q_hidden = k_hidden = v_hidden = inputs[..., 0, :]
            else:
                q_hidden, k_hidden, v_hidden = inputs.unbind(-2)
        if messages is not None:
            q_message, k_message, v_message, mlp_message = messages.unbind(-2)
            q_hidden = self.q_message_gate(q_hidden, q_message)
            k_hidden = self.k_message_gate(k_hidden, k_message)
            v_hidden = self.v_message_gate(v_hidden, v_message)
        if self.is_sliding:
            q_hidden = self.input_layernorm(q_hidden)
            k_hidden = self.input_layernorm(k_hidden)
            v_hidden = self.input_layernorm(v_hidden)
        else:
            q_hidden = self.q_input_layernorm(q_hidden)
            k_hidden = self.k_input_layernorm(k_hidden)
            v_hidden = self.v_input_layernorm(v_hidden)
        attn_out = self.self_attn(
            q_hidden.to(dtype),
            k_hidden.to(dtype),
            v_hidden.to(dtype),
            position_ids,
            cu_seqlens,
            doc_ids,
            block_masks["sliding" if self.is_sliding else "full"],
        )
        attn_out = self.post_attention_convolution(attn_out, cu_seqlens, doc_ids)
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
            residual = attn_out.float()
        else:
            residual = block_residual.float() + attn_out.float()
        mlp_input = self.mlp_input(mlp_query, block_states, residual)[..., 0, :]
        if messages is not None:
            mlp_input = self.mlp_message_gate(mlp_input, mlp_message)
        mlp_out = self.mlp(self.post_attention_layernorm(mlp_input).to(dtype), routed_experts=routed_experts)
        mlp_out = self.post_mlp_convolution(mlp_out, cu_seqlens, doc_ids)
        return (residual + mlp_out.float()).to(dtype), block_states


class Inkling2LoopModel(Inkling2Model):
    decoder_layer_class = Inkling2LoopDecoderLayer

    @property
    def num_execution_layers(self):
        config = self.config
        return config.num_hidden_layers + (config.loop_layer_index_end - config.loop_layer_index_start) * (
            config.loop_num_passes - 1
        )

    def __init__(self, config):
        super().__init__(config)
        self.attn_res.float()
        self.message_input = MessageInput(config)
        self.norm.preserve_input_dtypes = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MessageInput):
            nn.init.zeros_(module.query)

    def run_layers(self, hidden_states, block_states, position_ids, cu_seqlens, doc_ids, block_masks, routed_experts):
        start, end = self.config.loop_layer_index_start, self.config.loop_layer_index_end
        visits = self.num_execution_layers
        if routed_experts is not None and routed_experts.shape[-2] != visits:
            raise ValueError(f"Loop router replay requires {visits} execution visits, got {routed_experts.shape[-2]}")
        visit = 0

        def run(index, hidden, blocks, messages=None):
            nonlocal visit
            # Checkpoint recomputation can materialize saved route views. Give
            # compiled forward/backward the same dense layout at the boundary.
            routes = routed_experts[..., visit, :].contiguous() if routed_experts is not None else None
            visit += 1
            return self.layers[index](
                hidden,
                blocks,
                attn_query=self.attn_res.attn_queries[index - 1] if index else None,
                mlp_query=self.attn_res.mlp_queries[index],
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                doc_ids=doc_ids,
                block_masks=block_masks,
                messages=messages,
                routed_experts=routes,
            )

        for index in range(start):
            hidden_states, block_states = run(index, hidden_states, block_states)
        pre_hidden, pre_blocks = hidden_states, block_states
        for pass_index in range(self.config.loop_num_passes):
            if pass_index:
                previous_hidden = hidden_states
                previous_blocks = block_states[..., pre_blocks.shape[-2] + 1 :, :]
                hidden_states, block_states = pre_hidden, pre_blocks
            for index in range(start, end):
                messages = None
                if pass_index:
                    messages = self.message_input.forward(
                        self.message_input.query[index - start], previous_blocks, previous_hidden
                    )
                hidden_states, block_states = run(index, hidden_states, block_states, messages)
        for index in range(end, self.config.num_hidden_layers):
            hidden_states, block_states = run(index, hidden_states, block_states)
        return hidden_states, block_states


class Inkling2LoopForCausalLM(Inkling2ForCausalLM):
    text_config_class = Inkling2LoopTextConfig
    model_class = Inkling2LoopModel
    lora_export_target_modules = ["wo_ud"]

    def convert_adapter_to_hf(self, state_dict):
        """Export output-projection adapters for this fork's unrolled inference model."""
        start, end = self.config.loop_layer_index_start, self.config.loop_layer_index_end
        visits = list(range(start)) + list(range(start, end)) * self.config.loop_num_passes
        visits += list(range(end, self.config.num_hidden_layers))
        expanded = {}
        for name, value in state_dict.items():
            match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.o_proj\.(lora_[AB]\.weight)", name)
            if match is None:
                raise ValueError("Inkling loop LoRA currently supports only self_attn.o_proj")
            for visit, original in enumerate(visits):
                if original == int(match[1]):
                    expanded[f"model.layers.{visit}.attn.wo_ud.{match[2]}"] = value
        return expanded

    def fp32_modules(self):
        modules = [self.model.attn_res, self.model.message_input]
        for layer in self.model.layers:
            modules.extend(
                [layer.mlp.router.gate, layer.post_attention_convolution.conv1d, layer.post_mlp_convolution.conv1d]
            )
            if layer.is_sliding:
                modules.extend([layer.self_attn.k_sconv.conv1d, layer.self_attn.v_sconv.conv1d])
            if hasattr(layer, "q_message_gate"):
                for name in ("q", "k", "v", "mlp"):
                    gate = getattr(layer, f"{name}_message_gate")
                    modules.extend([gate.forget_gate_proj, gate.message_gate_proj])
        return modules

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name):
        return (
            super().keep_in_fp32_for_weight_transfer(name)
            or "message_input.query" in name
            or "forget_gate_proj" in name
            or "message_gate_proj" in name
            or "mlp.router.gate.weight" in name
            or ".conv1d.weight" in name
        )
