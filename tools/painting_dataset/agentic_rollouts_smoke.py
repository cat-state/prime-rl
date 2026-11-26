"""CPU-only concurrency checks for batched episode turn barriers and failure cleanup."""
import argparse
import base64
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from tools.painting_dataset.agentic_episode import EpisodeConfig
from tools.painting_dataset.agentic_rollouts import collect_rollouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-policy", choices=["full-v1", "bounded-v2"], default="full-v1")
    parser.add_argument("--turns", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    target = args.output / "target.png"
    Image.new("RGB", (8, 8), "red").save(target)

    def encode(messages, directory):
        images = [p for m in messages if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image"]
        features = directory / "fake.json"
        features.write_text("{}")
        return {"features": str(features), "input_ids": [1]*len(images)+[2],
                "mm_token_type_ids": [1]*len(images)+[0],
                "image_spans": [{"start": i, "count": 1, "image": p["path"], "image_sha256": p["sha256"]}
                                for i, p in enumerate(images)]}

    class Generator:
        policy = "test"
        def generate_batch(self, jobs):
            responses = []
            for e, r in jobs:
                shape = [len(e["input_ids"])+1, 48, 8]
                responses.append({"prompt_token_ids": e["input_ids"], "completion_ids": [4, 5],
                    "completion_logprobs": [-.1, -.2], "raw_output": "```wgsl\nfn paint(uv:vec2f)->vec3f {return vec3f(1.);}\n```",
                    "finish_reason": "stop", "temperature": r["temperature"], "policy": self.policy,
                    "routed_experts": {"start": 0, "shape": shape, "dtype": "uint8",
                        "data": base64.b64encode(np.zeros(shape, dtype=np.uint8).tobytes()).decode()}})
            return responses

    @contextmanager
    def factory(workspace):
        class Renderer:
            def render(self, source, directory, width, height):
                image = directory / "result.png"
                Image.new("RGB", (width, height), "red").save(image)
                return {"ok": True, "image": str(image)}
        yield Renderer()

    jobs = [{"target": str(target), "seed": 100+i*10} for i in range(4)]
    result = collect_rollouts(jobs=jobs, output=args.output, encoder=encode, generator=Generator(), batch_size=2,
        config=EpisodeConfig(turns=args.turns, history_policy=args.history_policy), renderer_factory=factory, reward_fn=lambda *_: {"diagnostic": 1})
    assert len(set(result["episodes"])) == 4
    assert [b["size"] for b in result["generation_batches"]] == [2]*(2*args.turns)
    for ep in result["episodes"]:
        metadata = json.loads((Path(ep)/"turn-1/encoded.json").read_text())
        assert len(metadata["image_spans"]) == 2
        if args.history_policy == "bounded-v2" and args.turns == 3:
            last = Path(ep)/"turn-2"
            metadata = json.loads((last/"encoded.json").read_text())
            assert len(metadata["image_spans"]) == 2
            training = json.loads((last/"training.json").read_text())
            n = len(metadata["input_ids"])
            assert training["loss_mask"] == [False]*n + [True]*2
            assert training["routed_experts"]["shape"] == [n+1, 48, 8]
            assert len(json.loads((last/"messages.json").read_text())) == 3
    bad = args.output / "failure"
    bad.mkdir()
    def broken_encoder(messages, directory):
        raise ValueError("injected encoder failure")
    try:
        collect_rollouts(jobs=jobs, output=bad, encoder=broken_encoder, generator=Generator(), batch_size=2,
            config=EpisodeConfig(turns=2), renderer_factory=factory, reward_fn=lambda *_: {})
    except ValueError as error:
        assert str(error) == "injected encoder failure"
    else:
        raise AssertionError("Failure must propagate without deadlocking workers")
    print(json.dumps({"passed": True, "episodes": 4, "turns": args.turns, "history_policy": args.history_policy, "failure_cleanup": True}))


if __name__ == "__main__":
    main()
