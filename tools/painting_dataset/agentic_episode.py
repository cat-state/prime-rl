"""Isolated, bounded WGSL revision episodes with replayable per-turn evidence.

Providers are trusted application code. Model actions contain only WGSL source;
no model-provided paths, commands, or renderer arguments are executed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2))


def confined(path: str | Path, root: Path) -> Path:
    result = Path(path).resolve(strict=True)
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Artifact escaped episode workspace")
    return result


def extract_action(response: dict, entrypoint: str) -> str:
    if response["finish_reason"] != "stop":
        raise ValueError("Generation truncated before complete action")
    text = response["raw_output"]
    if "<|content_text|>" in text:
        text = text.split("<|content_text|>", 1)[1]
    text = text.split("<|end_message|>", 1)[0]
    match = re.fullmatch(r"\s*```(?:wgsl)?\s*\n(.*?)\n```\s*", text, re.S)
    if not match:
        raise ValueError("Expected exactly one final WGSL code block")
    source = match[1]
    if len(source.encode()) > 40000 or "@" in source or not re.search(rf"\bfn\s+{re.escape(entrypoint)}\s*\(", source):
        raise ValueError("Shader violates source budget or renderer interface")
    return source


def validate_turn(response: dict, encoded: dict) -> dict:
    """Retain independent prompt/completion replay, including multiple image spans."""
    prompt = response["prompt_token_ids"]
    completion = response["completion_ids"]
    probs = response["completion_logprobs"]
    if prompt != encoded["input_ids"] or not completion or len(probs) != len(completion):
        raise ValueError("Generation token evidence does not match encoded prompt")
    if not np.isfinite(probs).all():
        raise ValueError("Nonfinite sampled logprobs")
    route = response["routed_experts"]
    shape = [len(prompt) + len(completion) - 1, 48, 8]
    if route["start"] != 0 or route["shape"] != shape:
        raise ValueError("Incomplete per-visit replay routes")
    dtype = np.dtype(route["dtype"])
    if dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("int32"), np.dtype("int64")):
        raise ValueError("Unsupported route dtype")
    raw = base64.b64decode(route["data"], validate=True)
    if len(raw) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("Route payload length mismatch")
    types = encoded["mm_token_type_ids"]
    if len(types) != len(prompt) or not set(types).issubset({0, 1}):
        raise ValueError("Invalid image mask")
    covered = np.zeros(len(prompt), dtype=np.int32)
    for span in encoded["image_spans"]:
        start, count = span["start"], span["count"]
        if start < 0 or count <= 0 or start + count > len(prompt):
            raise ValueError("Invalid image span")
        covered[start:start + count] += 1
        if digest(Path(span["image"])) != span["image_sha256"]:
            raise ValueError("Image changed after encoding")
    if not np.array_equal(covered, types):
        raise ValueError("Image spans do not exactly partition image mask")
    return {"token_ids": prompt + completion, "loss_mask": [False] * len(prompt) + [True] * len(completion),
            "logprobs": [0.] * len(prompt) + probs, "mm_token_type_ids": types + [0] * len(completion),
            "image_spans": encoded["image_spans"], "routed_experts": route,
            "temperature": response["temperature"], "policy": response["policy"],
            "adapter_sha256": response.get("adapter_sha256"),
            "note": "Independent teacher-forced turn; context tokens masked. Advantage assigned after episode scoring."}


class Renderer(Protocol):
    def render(self, source: str, directory: Path, width: int, height: int) -> dict: ...


@dataclass
class EpisodeConfig:
    turns: int = 3
    max_tokens: int = 1536
    temperature: float = .7
    resolution: int = 256
    entrypoint: str = "paint"


def run_episode(*, root: Path, target: Path, contract: str, encoder: Callable,
                generator: Callable, renderer_factory: Callable, reward_fn: Callable,
                seed: int, policy: str, config: EpisodeConfig | None = None) -> Path:
    """Encoder(messages,dir)->metadata; generator(metadata,request)->full response.

    renderer_factory(workspace) is a context manager owning its process/socket.
    Encoder must save frozen multimodal features under its supplied directory.
    Reward function(target,rendered) is separate from orchestration and returns
    named metrics, retaining the reward definition/version alongside results.
    """
    config = config or EpisodeConfig()
    if config.turns not in (2, 3) or config.temperature <= 0:
        raise ValueError("Use two or three turns and positive sampling temperature")
    root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="ep-", dir=root)).resolve()
    copied = workspace / "target.png"
    with Image.open(target) as image:
        image.convert("RGB").save(copied)
        width, height = image.size
    scale = config.resolution / max(width, height)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    messages = [{"role": "user", "content": [{"type": "image", "path": str(copied), "sha256": digest(copied)},
                 {"type": "text", "text": "Reconstruct this painting with WGSL. Render feedback will follow.\n" + contract}]}]
    state = {"seed": seed, "policy": policy, "target_sha256": digest(copied), "config": vars(config), "turns": []}
    write_json(workspace / "episode.json", state)
    with renderer_factory(workspace) as renderer:
        for turn in range(config.turns):
            directory = workspace / f"turn-{turn}"
            directory.mkdir()
            write_json(directory / "messages.json", messages)
            encoded = encoder(messages, directory)
            confined(encoded["features"], directory)
            for span in encoded["image_spans"]:
                confined(span["image"], workspace)
            write_json(directory / "encoded.json", encoded)
            request = {"seed": seed + turn, "policy": policy, "max_tokens": config.max_tokens,
                       "temperature": config.temperature}
            response = generator(encoded, request)
            write_json(directory / "response.json", response)
            if response["policy"] != policy or response["temperature"] != config.temperature:
                raise ValueError("Generation policy or temperature changed")
            if turn == 0:
                state["adapter_sha256"] = response.get("adapter_sha256")
            elif response.get("adapter_sha256") != state["adapter_sha256"]:
                raise ValueError("Adapter changed within episode")
            training = validate_turn(response, encoded)
            write_json(directory / "training.json", training)
            record = {"turn": turn, "valid": False, "tokens": len(response["completion_ids"])}
            try:
                source = extract_action(response, config.entrypoint)
                (directory / "action.wgsl").write_text(source)
                rendered = renderer.render(source, directory, width, height)
                write_json(directory / "render.json", rendered)
                if not rendered.get("ok"):
                    raise ValueError(str(rendered.get("error", "Renderer failed"))[:2000])
                image_path = confined(rendered["image"], directory)
                with Image.open(image_path) as image:
                    if image.size != (width, height):
                        raise ValueError("Unexpected render size")
                    image.verify()
                record.update(valid=True, image=str(image_path), metrics=reward_fn(copied, image_path))
                feedback = [{"type": "image", "path": str(image_path), "sha256": digest(image_path)},
                            {"type": "text", "text": "This is your rendered result. Compare it with the target and improve the complete WGSL program."}]
            except (ValueError, OSError, TimeoutError) as error:
                record["error"] = str(error)[:2000]
                feedback = [{"type": "text", "text": "Render failed: " + record["error"] + "\nRevise the complete WGSL program."}]
            state["turns"].append(record)
            write_json(workspace / "episode.json", state)
            messages.append({"role": "assistant", "content": response["raw_output"]})
            messages.append({"role": "user", "content": feedback})
    state["complete"] = True
    write_json(workspace / "episode.json", state)
    return workspace
