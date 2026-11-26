"""CPU contract smoke: isolated revisions, failure feedback and multi-image replay."""
import argparse
import base64
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from tools.painting_dataset.agentic_episode import confined, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    target = args.output / "input.png"
    Image.new("RGB", (16, 16), "red").save(target)

    def encode(messages, directory):
        images = [part for message in messages if isinstance(message["content"], list)
                  for part in message["content"] if part["type"] == "image"]
        features = directory / "fake-features.json"
        features.write_text("{}")
        return {"features": str(features), "input_ids": [1] * len(images) + [2, 3],
                "mm_token_type_ids": [1] * len(images) + [0, 0],
                "image_spans": [{"start": i, "count": 1, "image": item["path"],
                                 "image_sha256": item["sha256"]} for i, item in enumerate(images)]}

    calls = []
    def generate(encoded, request):
        calls.append(encoded)
        source = "invalid action" if len(calls) == 1 else "```wgsl\nfn paint(uv:vec2f)->vec3f {return vec3f(1.,0.,0.);}\n```"
        shape = [len(encoded["input_ids"]) + 1, 48, 8]
        return {"prompt_token_ids": encoded["input_ids"], "completion_ids": [4, 5],
                "completion_logprobs": [-.1, -.2], "raw_output": source, "finish_reason": "stop",
                "temperature": request["temperature"], "policy": request["policy"],
                "routed_experts": {"start": 0, "shape": shape, "dtype": "uint8",
                                   "data": base64.b64encode(np.zeros(shape, dtype=np.uint8).tobytes()).decode()}}

    class FakeRenderer:
        def render(self, source, directory, width, height):
            path = directory / "render.png"
            Image.new("RGB", (width, height), "red").save(path)
            return {"ok": True, "image": str(path)}

    @contextmanager
    def renderer_factory(workspace):
        yield FakeRenderer()

    episode = run_episode(root=args.output, target=target, contract="test paint interface", encoder=encode,
                          generator=generate, renderer_factory=renderer_factory,
                          reward_fn=lambda target, rendered: {"test_only": True}, seed=1, policy="test")
    state = json.loads((episode / "episode.json").read_text())
    assert state["complete"] and [r["valid"] for r in state["turns"]] == [False, True, True]
    assert len(calls[2]["image_spans"]) == 2
    evidence = json.loads((episode / "turn-2/training.json").read_text())
    assert sum(evidence["loss_mask"]) == 2 and len(evidence["image_spans"]) == 2
    import torch

    from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
    from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
    from tools.painting_dataset.agentic_batch import turn_to_microbatch

    directory = episode / "turn-2"
    encoded = json.loads((directory / "encoded.json").read_text())
    response = json.loads((directory / "response.json").read_text())
    records = []
    for i, span in enumerate(encoded["image_spans"]):
        path = directory / f"image-{i}.pt"
        torch.save({"image_features": torch.full((span["count"], 4), float(i + 1))}, path)
        records.append({"source_image_sha256": span["image_sha256"], "features": str(path)})
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"encoder_frozen": True, "records": records}))
    cache = FrozenImageCache(manifest, 4)
    batch = turn_to_microbatch(response, encoded, advantage=.5, expected_policy="test")
    tensors = prepare_micro_batch(DataLoader._micro_batch_to_tensor(None, batch), None, cache.materialize)
    assert tensors["mm_kwargs"]["image_features"].shape == (2, 4)
    assert tensors["mm_kwargs"]["image_token_mask"].sum() == 2
    assert tensors["loss_mask"].sum() == 2
    try:
        confined(target, episode)
    except ValueError:
        pass
    else:
        raise AssertionError("Cross-workspace artifact was accepted")
    print(json.dumps({"passed": True, "episode": str(episode), "gpu_used": False}))


if __name__ == "__main__":
    main()
