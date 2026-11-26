"""Opt-in storage sharing for BF16 Triton experts across loop visits.

Only weights alias. Each visit retains its module, Parameter objects, loaders,
router, kernel and cache identity. Other backends and expert parallelism are
deliberately unsupported until their weight transformations are validated.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

import torch
from vllm.platforms import current_platform

_groups = ContextVar("inkling_shared_expert_groups", default=None)
WEIGHTS = ("w13_weight", "w2_weight")


def _kernel_parameter(layer, name):
    # vLLM checkpoint reload stages one layer at a time on meta/new storage,
    # then copies into these saved Parameters to preserve CUDA graph addresses.
    # Check the actual runtime storage, not the temporary checkpoint staging.
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

    info = LAYERWISE_INFO.get(layer)
    if info is not None and info.kernel_tensors is not None:
        return info.kernel_tensors[0][name]
    return getattr(layer, name)


def validate_group(group):
    source = group[0]
    for layer in group:
        for name in WEIGHTS:
            a, b = _kernel_parameter(source, name), _kernel_parameter(layer, name)
            if (a.shape, a.stride(), a.dtype, a.device) != (b.shape, b.stride(), b.dtype, b.device):
                raise RuntimeError(f"Shared expert layout changed: {name}")
            if a.data_ptr() != b.data_ptr():
                raise RuntimeError(f"Shared expert storage detached: {name}")
            if source is not layer and a is b:
                raise RuntimeError("Visits must retain independent Parameter objects")


@contextmanager
def sharing_scope(enabled):
    groups = {} if enabled else None
    token = _groups.set(groups)
    try:
        yield groups
    finally:
        _groups.reset(token)


def share_visit_experts(layer, original):
    groups = _groups.get()
    if groups is None:
        return
    if not current_platform.is_cuda():
        raise ValueError("Inkling expert sharing currently requires NVIDIA CUDA")
    method = layer.quant_method
    backend = getattr(method, "unquantized_backend", None)
    if getattr(backend, "name", None) != "TRITON":
        raise ValueError("Inkling expert sharing requires the unquantized TRITON backend")
    parallel = layer.moe_config.moe_parallel_config
    if parallel.enable_eplb or parallel.ep_size != 1:
        raise ValueError("Inkling expert sharing does not support EP or EPLB")
    for name in WEIGHTS:
        parameter = getattr(layer, name)
        if parameter.dtype != torch.bfloat16 or parameter.device.type != "cuda" or not parameter.is_contiguous():
            raise ValueError("Inkling expert sharing requires contiguous CUDA BF16 weights")
    group = groups.setdefault(original, [])
    if group:
        source = group[0]
        for name in WEIGHTS:
            getattr(layer, name).data = getattr(source, name).data
    group.append(layer)
    validate_group(group)
    original_setup = method._setup_kernel

    def setup_shared_kernel(self, layer, w13, w2):
        validate_group(group)
        if layer is group[0]:
            # Triton CUDA leaves the contiguous checkpoint layout unchanged.
            # Assert that future dependency changes cannot silently detach it.
            original_setup(layer=layer, w13=w13, w2=w2)
        elif self.moe_kernel is None:
            # The canonical visit already processed the common weights. Each
            # later visit still needs its own kernel/routing metadata.
            if group[0].quant_method.moe_kernel is None:
                raise RuntimeError("Canonical shared experts must be processed first")
            self._init_moe_kernel(layer)
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)
        validate_group(group)

    method._setup_kernel = MethodType(setup_shared_kernel, method)
    original_process = method.process_weights_after_loading

    def process_shared_weights(self, layer):
        validate_group(group)
        original_process(layer)
        validate_group(group)

    method.process_weights_after_loading = MethodType(process_shared_weights, method)


def storage_report(groups):
    report = []
    for original, group in (groups or {}).items():
        validate_group(group)
        size = sum(getattr(group[0], name).numel() * getattr(group[0], name).element_size() for name in WEIGHTS)
        report.append({"original": original, "visits": len(group), "bytes_per_copy": size,
                       "saved_bytes": (len(group) - 1) * size})
    return report
