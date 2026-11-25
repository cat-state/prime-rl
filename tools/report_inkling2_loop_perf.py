"""Report loop-aware 6N model TFLOP/s from a trainer metrics file (not hardware FLOP counters)."""

import argparse
import json
from pathlib import Path

import torch

from prime_rl.trainer.models.base import count_active_params
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopConfig, Inkling2LoopForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    config = Inkling2LoopConfig.from_dict(json.loads((args.model_dir / "config.json").read_text()))
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(config)
    start, end = config.loop_layer_index_start, config.loop_layer_index_end
    core = list(model.model.layers[start:end].parameters())
    physical_active = count_active_params(model)
    effective_active = physical_active + (config.loop_num_passes - 1) * count_active_params(model, core)
    # Cross-pass controllers are not applied on the first core pass.
    effective_active -= sum(
        parameter.numel()
        for layer in model.model.layers[start:end]
        for stream in ("q", "k", "v", "mlp")
        for parameter in getattr(layer, f"{stream}_message_gate").parameters()
    )
    steps = []
    for line in args.metrics.read_text().splitlines():
        record = json.loads(line)
        if "perf/throughput" in record:
            tflops = 6 * effective_active * record["perf/throughput"] / 1e12
            steps.append(
                {
                    "step": record["step"],
                    "tokens_per_second_fwd_bwd": record["perf/throughput"],
                    "model_tflops_aggregate": tflops,
                    "model_tflops_per_gpu": tflops / args.gpus,
                    "peak_memory_gib": record["perf/peak_memory"],
                }
            )
    report = {
        "method": "6N approximation counting shared core weights once per execution visit and active MoE experts only",
        "limitations": "Excludes attention mixing, rematerialization, optimizer, communication and rollout time; not hardware-counter FLOPs or end-to-end RL throughput. Includes small non-matmul parameters in N.",
        "physical_active_parameters": physical_active,
        "effective_active_parameters_per_token": effective_active,
        "gpus": args.gpus,
        "steps": steps,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
