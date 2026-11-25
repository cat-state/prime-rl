"""Loop recurrence, packing, gradient flow, and release checkpoint conversion."""

import json

import pytest
import torch
from safetensors.torch import save_file

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopConfig, Inkling2LoopForCausalLM, Inkling2LoopTextConfig
from prime_rl.trainer.models.inkling2_loop.converting_inkling2_loop import DequantizeExperts, MessageQueries
from prime_rl.trainer.rl.warmup import warmup_routed_experts
from prime_rl.utils.weights import load_state_dict, load_state_dict_keys


def tiny_config(passes=3):
    return Inkling2LoopTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=8,
        eos_token_id=127,
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
        loop_num_passes=passes,
        loop_gate_rank=8,
    )


def test_message_query_roundtrip():
    queries = torch.arange(256, dtype=torch.float32).view(4, 64)
    kv = -queries
    state = {MessageQueries.hf: queries, MessageQueries.hf_kv: kv}
    op = MessageQueries()
    op.hf_to_prime(state)
    torch.testing.assert_close(state[op.prime][:, 0].float(), queries[::2])
    torch.testing.assert_close(state[op.prime][:, 3].float(), queries[1::2])
    op.prime_to_hf(state)
    torch.testing.assert_close(state[op.hf], queries)
    torch.testing.assert_close(state[op.hf_kv], kv)


def test_checkpoint_config_roundtrip():
    config = Inkling2LoopConfig(text_config=tiny_config().to_dict(), quantization_config={"quant_method": "modelopt"})
    exported = config.to_dict()
    assert "quantization_config" not in exported
    assert exported["architectures"] == ["Inkling2LoopForConditionalGeneration"]
    assert "loop_num_passes" not in exported["text_config"]
    restored = Inkling2LoopConfig.from_dict(exported)
    assert restored.to_dict() == exported


def test_checkpoint_index_excludes_separate_draft(tmp_path):
    weight = torch.ones(2)
    save_file({"model.llm.embed.weight": weight}, tmp_path / "model-1.safetensors")
    save_file({"fuse_weight": weight}, tmp_path / "mtp.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.llm.embed.weight": "model-1.safetensors"}})
    )
    assert load_state_dict_keys(tmp_path) == ["model.llm.embed.weight"]
    assert set(load_state_dict(tmp_path)) == {"model.llm.embed.weight"}


def test_quantized_experts():
    # Each byte has +0.5 in its low nibble and -1.0 in its high nibble.
    state = {
        "w": torch.full((2, 3, 8), 0xA1, dtype=torch.uint8),
        "w_scale": torch.full((2, 3, 1), 2.0),
        "w_amax": torch.tensor([3.0, 6.0]),
        "w_outer_scale_denominator": torch.tensor(3.0),
    }
    DequantizeExperts("w").hf_to_prime(state)
    expected = torch.tensor([1.0, -2.0] * 8).expand(2, 3, 16).clone()
    expected[1] *= 2
    torch.testing.assert_close(state["w"].float(), expected)
    assert set(state) == {"w"}
    state = {
        "w": torch.ones(2, 3, 32).to(torch.float8_e4m3fn),
        "w_scale": torch.full((2, 3, 1), 128, dtype=torch.uint8),
    }
    DequantizeExperts("w").hf_to_prime(state)
    torch.testing.assert_close(state["w"].float(), torch.full((2, 3, 32), 2.0))


@pytest.mark.gpu
def test_loop_restart_gradients_packing_and_conversion():
    torch.manual_seed(17)
    model = Inkling2LoopForCausalLM.from_config(tiny_config(), dtype=torch.bfloat16).cuda()
    warmup_routes = warmup_routed_experts(model, 16, torch.Generator().manual_seed(7))
    assert warmup_routes.shape == (1, 16, 16, 2)
    ids = torch.randint(0, 128, (1, 16), device="cuda")
    lengths = torch.tensor([16], device="cuda")
    # Zero gates make subsequent passes reproduce the first pass: simply continuing
    # the residual stack (instead of restarting it) fails this check.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "message_gate_proj" in name or "forget_gate_proj" in name:
                parameter.zero_()
        three = model.model(ids, seq_lens=lengths).last_hidden_state
        model.config.loop_num_passes = 1
        one = model.model(ids, seq_lens=lengths).last_hidden_state
    torch.testing.assert_close(three, one, atol=1e-3, rtol=1e-3)
    model.config.loop_num_passes = 3
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "message_gate_proj" in name or "forget_gate_proj" in name or name.endswith("message_input.query"):
                parameter.normal_(std=0.1)
    positions = torch.arange(8, device="cuda").repeat(2).unsqueeze(0)
    packed = model.model(ids, position_ids=positions, seq_lens=torch.tensor([8, 8], device="cuda")).last_hidden_state
    separate = torch.cat(
        [
            model.model(part, seq_lens=torch.tensor([8], device="cuda")).last_hidden_state
            for part in ids.split(8, dim=1)
        ],
        dim=1,
    )
    torch.testing.assert_close(packed, separate, atol=0.025, rtol=0.025)
    # Repeated physical layers must consume distinct execution-visit routes.
    routes = torch.stack([torch.tensor([i % 4, (i + 1) % 4]) for i in range(16)]).cuda()
    routes = routes.expand(1, 16, -1, -1).contiguous()
    captured = []

    def record_routes(module, args, output):
        captured.append(output[1].detach())

    handles = [layer.mlp.router.register_forward_hook(record_routes) for layer in model.model.layers]
    with torch.no_grad():
        replayed = model.model(ids, seq_lens=lengths, routed_experts=routes).last_hidden_state
    for handle in handles:
        handle.remove()
    assert len(captured) == 16
    for visit, selected in enumerate(captured):
        assert torch.equal(selected.reshape(1, 16, 2), routes[:, :, visit])
    assert torch.isfinite(replayed).all()
    with pytest.raises(ValueError, match="16 execution visits"):
        model.model(ids, seq_lens=lengths, routed_experts=routes[:, :, :8])
    loss = (packed.float() * torch.randn_like(packed.float())).mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        if "message" in name or "model.layers.2.self_attn.q_proj" in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original = {name: value.clone() for name, value in state.items()}
    ops = model.conversion_chain(model.config)
    apply_prime_to_hf(state, ops)
    apply_hf_to_prime(state, ops)
    assert state.keys() == original.keys()
    for name in state:
        torch.testing.assert_close(state[name], original[name], atol=1e-6, rtol=1e-5)
