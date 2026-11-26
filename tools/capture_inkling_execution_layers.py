"""Diagnostic capture outside compiled layer calls; never installed in production."""

import inspect
import textwrap

import torch

from prime_rl.trainer.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopModel


def install_capture():
    """Sample token positions without changing tensors passed between layers."""
    captured = {}

    def capture(visit, physical_index, result):
        for name, value in zip(("hidden", "blocks"), result, strict=True):
            # Copies occur after the compiled/FSDP layer returns. No module hooks
            # or instrumentation inside the compiled graph are installed.
            indices = torch.linspace(0, value.shape[1] - 1, min(64, value.shape[1]),
                                     device=value.device).round().long().unique()
            key = f"visit{visit - 1:02d}_layer{physical_index:02d}_{name}"
            captured[key] = value.detach().index_select(1, indices).cpu().contiguous()
            captured[key + "_indices"] = indices.cpu().contiguous()

    source = textwrap.dedent(inspect.getsource(Inkling2LoopModel.run_layers))
    before = "return self.layers[index]("
    assert source.count(before) == 1
    source = source.replace(before, "result = self.layers[index](", 1)
    boundary = "        )\n\n    for index in range(start):"
    assert source.count(boundary) == 1
    source = source.replace(boundary,
        "        )\n        capture(visit, index, result)\n        return result\n\n    for index in range(start):", 1)
    namespace = dict(Inkling2LoopModel.run_layers.__globals__, capture=capture)
    exec(compile(source, "<diagnostic_execution_capture>", "exec"), namespace)
    Inkling2LoopModel.run_layers = namespace["run_layers"]
    return captured
