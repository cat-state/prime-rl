"""Run a real two/three-turn painting episode with separately placed encoder/inference."""
import argparse
import json
from pathlib import Path

import torch

from tools.painting_dataset.agentic_episode import EpisodeConfig, run_episode
from tools.painting_dataset.agentic_providers import FrozenEncoderClient, SharedInferenceGenerator
from tools.painting_dataset.painting_reward import score

CONTRACT = """Return exactly one complete WGSL code block in your final answer.
Implement fn paint(uv:vec2f)->vec3f returning display RGB in [0,1].
UV (0,0) is top left and (1,1) bottom right. Each pixel is sampled at its center.
This is a direct 2D painting, not a distance field or 3D material-ID shader.
No bindings, entry points, attributes, textures, arrays, global variables, loops,
recursion, or external inputs. Helper functions, local variables, conditionals,
and WGSL math are allowed. Do not redefine paint more than once.
Use a compact composition with colored shapes and smooth transitions. Complete
the program within the answer budget. You will see the rendering and may revise.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", type=Path, help="Extend a completed shorter episode with matching settings")
    parser.add_argument("--encoder-gpu", default="3")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--kv-cache-gib", type=int, default=24)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--final-only", action="store_true", help="Explicitly prefill the model final-channel header")
    parser.add_argument("--instruction", default="", help="Additional task-specific instruction, retained in messages")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume is not None)
    with FrozenEncoderClient(args.output, args.encoder_gpu, final_only=args.final_only) as encoder:
        generator = SharedInferenceGenerator(max_model_len=args.max_model_len, kv_cache_gib=args.kv_cache_gib)
        from tools.painting_dataset.rgb_renderer import factory
        episode = run_episode(root=args.output, target=args.target, contract=CONTRACT + "\n" + args.instruction,
                              encoder=encoder, generator=generator, renderer_factory=factory,
                              reward_fn=score, seed=401, policy="base",
                              config=EpisodeConfig(turns=args.turns, max_tokens=args.max_tokens), resume=args.resume)
        # Exercise the actual multi-image Prime bridge and frozen cache, without
        # an optimizer step or loading a second trainer model.
        from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
        from prime_rl.trainer.rl.data import DataLoader, prepare_micro_batch
        from tools.painting_dataset.agentic_batch import turn_to_microbatch
        checks = []
        for directory in sorted(episode.glob("turn-*")):
            encoded = json.loads((directory / "encoded.json").read_text())
            response = json.loads((directory / "response.json").read_text())
            hidden = torch.load(encoded["features"], weights_only=True, map_location="cpu")["prompt_embeds"].shape[-1]
            cache = FrozenImageCache(Path(encoded["feature_manifest"]), hidden)
            batch = turn_to_microbatch(response, encoded, advantage=1., expected_policy="base")
            tensors = prepare_micro_batch(DataLoader._micro_batch_to_tensor(None, batch), None, cache.materialize)
            assert tensors["loss_mask"].sum() == len(response["completion_ids"])
            assert tensors["mm_kwargs"]["image_token_mask"].sum() == sum(s["count"] for s in encoded["image_spans"])
            checks.append({"turn": directory.name, "images": len(encoded["image_spans"]),
                           "tokens": len(response["completion_ids"]), "replay_materialization_passed": True})
        (episode / "replay-checks.json").write_text(json.dumps(checks, indent=2))
        print(json.dumps({"episode": str(episode), "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
