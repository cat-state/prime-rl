# Inkling 2 Mini (text)

PrimeRL supports the shared-weight prelude/core/coda loop in
`thinkingmachines/Inkling-2-Mini`, including Q/K/V/MLP cross-pass messages,
packed documents, checkpoint conversion, and BF16 inference weight updates.
Training reuses the same core parameters on every pass. Inference unrolls the
visits so each pass owns independent KV and short-convolution caches; its core
weights are duplicated and refreshed from the shared trainer weights on reload.

This is an experimental eager-mode integration. Use `compile = "None"`, `ep = 1`,
`cp = 1`, and enable `trainer.enable_router_replay` for RL. Routes are recorded
in execution order: prelude, every core pass, then coda (48 visits for Mini).
Vision, speculative decoding,
variable-pass overrides, and context parallelism are not supported. The SiTU
caps must match the release `(4, 25)`. Expert parallelism and a full RL run have
not been validated. Compiled FSDP matches replicated training when both use
FP32 masters and activation checkpointing in the tiny distributed check.
Comparison against a reference without checkpointing still exceeds the gradient
criterion; compiled training remains experimental pending downstream validation.

Output-projection LoRA is available experimentally with trainer targets
`["o_proj"]` and inference `lora_target_modules = ["wo_ud"]`. Set
`INKLING_MULTIMEM_AR=0` for vLLM. The shared training adapters are expanded to
execution visits when broadcast; these exported adapters target this fork's
inference layout and are not portable to the standard Transformers model.
Other adapter targets are rejected. See `configs/debug/inkling2-loop-rl.toml`
for the two-trainer/two-inference integration configuration.

## Install and infer

```bash
uv sync --all-extras
uv run python tools/prepare_inkling2_loop.py \
  thinkingmachines/Inkling-2-Mini /path/to/inkling2-mini-bf16 \
  --revision c9d028302cd0f035fdf8e7e91f089a4ce94d2195

CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve /path/to/inkling2-mini-bf16 \
  --served-model-name inkling-2-mini \
  --tensor-parallel-size 2 --max-model-len 2048 \
  --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 4294967296 \
  --enforce-eager --disable-flashinfer-autotune \
  --reasoning-parser inkling2 --host 127.0.0.1
```

The export dequantizes NVFP4/MXFP8 experts to BF16 and preserves the FP32 router,
convolution, controller, and residual-query parameters used by the trainer.
It takes approximately 52 GB of storage. The checkpoint index excludes the
separately packaged draft model. The output config contains text-only model
metadata without the original quantization configuration. Keep the released
checkpoint as the canonical source; the BF16 export is reproducible.

Two 96 GB GPUs accommodate the unrolled BF16 inference weights and a small
cache. The original quantized checkpoint cannot be passed directly to this
BF16 inference implementation. The ordinary PrimeRL weight-transfer path
accepts the exported weights and fans core weights out to all visits.

A trainer model configuration should contain:

```toml
[model]
name = "thinkingmachines/Inkling-2-Mini"
compile = "None"
ep = 1
cp = 1
conversion_dir = "/path/with/space/for/conversion"
```

This fragment is not a complete training run. Select the dataset, optimizer,
batch sizes, offloading, and inference/trainer GPU split separately. Loading the
release creates a BF16 conversion cache; budget storage and host RAM for it.

## Validation

```bash
uv run pytest tests/unit/train/models/test_inkling2_loop.py
uv run pytest tests/unit/train/models/test_scratch_init.py -k inkling
uv run pytest tests/unit/train/models/test_vllm_parity.py -k inkling2
uv run torchrun --standalone --nproc-per-node=2 \
  tools/validate_inkling2_loop.py /path/to/results
uv run torchrun --standalone --nproc-per-node=2 \
  tools/validate_inkling2_loop.py /path/to/compiled-results \
  --compile --fp32-reference --checkpoint-reference
```

The tests cover loop restart semantics, nonzero cross-pass gradients, packed
sequence isolation, NVFP4/MXFP8 decoding, export/import round trips, initialization,
and checkpoint-index selection. The distributed check uses real PrimeRL FSDP,
activation checkpointing, and an optimizer step. On two SM120 GPUs its eager loss
matched the unsharded loss and its worst normalized gradient error was 0.00960
against a 0.02 limit.

The inference test loads trainer weights through PrimeRL's layerwise reload
path, runs sixteen layer visits with nonzero message controllers, and checks
prompt log probabilities across sliding-window boundaries. On SM120 it measured
maximum log-probability error 0.0802 and mismatch KL 1.08e-4, with limits 0.2 and
5e-4. It uses per-visit router replay with ordinary routing scores, verifies the
replayed expert IDs exactly, and retains an unrestricted-routing diagnostic.
Near-tie routing can amplify rounding differences substantially;
these results establish loop/cache/weight-transfer consistency, not general
rollout/trainer equivalence for every prompt or training stability.
