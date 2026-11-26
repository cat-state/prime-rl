"""Convert one multimodal revision turn into a Prime replay microbatch."""
import base64
from pathlib import Path

import numpy as np

from prime_rl.orchestrator.batch import prepare_sample
from prime_rl.transports.batch.types import MMImageRef, MMRefs, TrainingSample
from prime_rl.transports.payload import _encode_routed_experts
from tools.painting_dataset.agentic_episode import validate_turn


def turn_to_microbatch(response, encoded, *, advantage, expected_policy):
    if response["policy"] != expected_policy or not np.isfinite(advantage):
        raise ValueError("Unexpected policy or nonfinite advantage")
    temperature = response["temperature"]
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Replay requires positive sampling temperature")
    evidence = validate_turn(response, encoded)
    tokens = np.asarray(evidence["token_ids"], dtype=np.int32)
    mask = np.asarray(evidence["loss_mask"], dtype=np.bool_)
    refs = [MMImageRef("data:image/png;base64," + base64.b64encode(Path(span["image"]).read_bytes()).decode(),
                       span["start"], span["count"]) for span in evidence["image_spans"]]
    sample = TrainingSample(token_ids=tokens, mask=mask, logprobs=np.asarray(evidence["logprobs"], dtype=np.float32),
                            temperatures=np.full(len(tokens), temperature, dtype=np.float32),
                            env_name="painting-agentic-frozen-images",
                            advantages=np.where(mask, advantage, 0).astype(np.float32),
                            mm_token_type_ids=np.asarray(evidence["mm_token_type_ids"], dtype=np.int32),
                            mm_refs=MMRefs(images=refs))
    route = evidence["routed_experts"]
    routes = np.frombuffer(base64.b64decode(route["data"], validate=True), dtype=np.dtype(route["dtype"])).reshape(route["shape"])
    batch = prepare_sample(sample, len(tokens))
    batch.routed_experts = _encode_routed_experts(routes, len(tokens))
    return batch
