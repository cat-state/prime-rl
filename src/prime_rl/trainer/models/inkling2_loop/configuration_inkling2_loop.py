"""Text-only training configuration for Inkling 2 Mini and variable-loop checkpoints."""

from prime_rl.trainer.models.inkling2.configuration_inkling2 import Inkling2TextConfig


class Inkling2LoopTextConfig(Inkling2TextConfig):
    model_type = "inkling2_loop_text"

    def __init__(
        self,
        loop_config=None,
        loop_layer_index_start=6,
        loop_layer_index_end=18,
        loop_num_passes=3,
        loop_gate_rank=64,
        **kwargs,
    ):
        loop = loop_config or {}
        self.loop_layer_index_start = loop.get("core_start", loop_layer_index_start)
        self.loop_layer_index_end = loop.get("core_end", loop_layer_index_end)
        self.loop_num_passes = loop.get("num_passes", loop_num_passes)
        self.loop_gate_rank = loop.get("gate_rank", loop_gate_rank)
        self.loop_config = {
            "core_start": self.loop_layer_index_start,
            "core_end": self.loop_layer_index_end,
            "num_passes": self.loop_num_passes,
            "gate_rank": self.loop_gate_rank,
            "is_varloop": loop.get("is_varloop", False),
        }
        defaults = dict(
            hidden_size=2304,
            num_hidden_layers=24,
            num_attention_heads=24,
            swa_num_attention_heads=24,
            n_routed_experts=128,
            moe_intermediate_size=1152,
            log_scaling_n_floor=8192,
        )
        defaults.update(kwargs)
        if "intermediate_size" in defaults:
            defaults["moe_intermediate_size"] = defaults.pop("intermediate_size")
        if "conv_kernel_size" in defaults:
            defaults["sconv_kernel_size"] = defaults.pop("conv_kernel_size")
        defaults.pop("model_type", None)
        super().__init__(**defaults)
        if (self.mlp_gate_softcap, self.mlp_up_softcap) != (4.0, 25.0):
            raise ValueError("Loop training currently supports the release SiTU caps (4, 25)")
        start, end = self.loop_layer_index_start, self.loop_layer_index_end
        if not 0 < start < end <= self.num_hidden_layers:
            raise ValueError("Loop bounds must satisfy 0 < start < end <= num_hidden_layers")
        if start % self.n_layers_per_attn_res_block or end % self.n_layers_per_attn_res_block:
            raise ValueError("Loop bounds must align with attention-residual blocks")
        if self.loop_num_passes < 1 or self.loop_gate_rank < 1:
            raise ValueError("Loop passes and gate rank must be positive")
        if 0 not in self.local_layer_ids:
            raise ValueError("The first layer must use sliding attention")


class Inkling2LoopConfig(Inkling2LoopTextConfig):
    model_type = "inkling2_loop"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        merged = {**(text_config or {}), **kwargs}
        for key in (
            "torch_dtype",
            "model_max_length",
            "mtp_config",
            "architectures",
            "quantization_config",
            "transformers_version",
        ):
            merged.pop(key, None)
        super().__init__(**merged)

    def to_dict(self):
        text = super().to_dict()
        text["model_type"] = "inkling2_model"
        text["intermediate_size"] = text.pop("moe_intermediate_size")
        # The checkpoint's fixed pass count lives in loop_config; a separate
        # loop_num_passes is a variable-loop inference override.
        for key in ("loop_num_passes", "loop_layer_index_start", "loop_layer_index_end", "loop_gate_rank"):
            text.pop(key, None)
        return {
            "model_type": self.model_type,
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "text_config": text,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
