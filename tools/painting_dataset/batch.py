"""Convert one trusted cached-image HTTP rollout into Prime's MicroBatch format.

The caller supplies scored advantage and actual sampling temperature. This does
not assign rewards, normalize groups, or send batches to a running trainer.
"""

import base64
import hashlib

import numpy as np

from prime_rl.orchestrator.batch import prepare_sample
from prime_rl.transports.batch.types import MMImageRef, MMRefs, TrainingSample
from prime_rl.transports.payload import _encode_routed_experts


def rollout_to_microbatch(response, image_bytes, *, temperature, advantage, expected_policy):
    if response["policy"] != expected_policy:
        raise ValueError("Rollout policy does not match requested training policy")
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(advantage):
        raise ValueError("Training requires positive sampling temperature and finite advantage")
    if hashlib.sha256(image_bytes).hexdigest() != response["source_image_sha256"]:
        raise ValueError("Image bytes do not match the rollout")
    prompt = np.asarray(response["prompt_token_ids"], dtype=np.int32)
    completion = np.asarray(response["completion_ids"], dtype=np.int32)
    logprobs = np.asarray(response["completion_logprobs"], dtype=np.float32)
    types = np.asarray(response["mm_token_type_ids"], dtype=np.int32)
    if not len(completion) or len(logprobs) != len(completion) or not np.isfinite(logprobs).all():
        raise ValueError("Invalid completion evidence")
    if len(types) != len(prompt) or not np.isin(types, [0, 1]).all():
        raise ValueError("Invalid image token metadata")
    positions = np.flatnonzero(types)
    if not len(positions) or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError("This bridge requires exactly one contiguous image span")
    tokens = np.concatenate((prompt, completion))
    route = response["routed_experts"]
    if route["start"] != 0 or route["shape"] != [len(tokens) - 1, 48, 8]:
        raise ValueError("Incomplete router coverage")
    dtype = np.dtype(route["dtype"])
    if dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("int32"), np.dtype("int64")):
        raise ValueError("Unsupported router dtype")
    routes = np.frombuffer(base64.b64decode(route["data"], validate=True), dtype=dtype).reshape(route["shape"])
    mask = np.arange(len(tokens)) >= len(prompt)
    sample = TrainingSample(
        token_ids=tokens, mask=mask,
        logprobs=np.concatenate((np.zeros(len(prompt), dtype=np.float32), logprobs)),
        temperatures=np.full(len(tokens), temperature, dtype=np.float32), env_name="painting-cached-image",
        advantages=np.where(mask, advantage, 0).astype(np.float32),
        mm_token_type_ids=np.concatenate((types, np.zeros(len(completion), dtype=np.int32))),
        mm_refs=MMRefs(images=[MMImageRef("data:image/jpeg;base64," + base64.b64encode(image_bytes).decode(),
                                       int(positions[0]), len(positions))]),
    )
    batch = prepare_sample(sample, len(tokens))
    # Prime's existing convention pads only the unsampled final-token route row.
    batch.routed_experts = _encode_routed_experts(routes, len(tokens))
    return batch
