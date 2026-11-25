"""Materialize packed frozen image features from an explicit trusted cache.

This boundary does not enable native image serving or register VLM support. The
caller must supply the actual packed sequence length and identical encoder cache
to rollout inference. Text embeddings are never loaded from the cache.
"""

import base64
import hashlib
import json
from pathlib import Path

import torch

from prime_rl.transports.batch.types import MMRefs


class FrozenImageCache:
    def __init__(self, manifest: Path, hidden_size: int):
        metadata = json.loads(manifest.read_text())
        if metadata.get("encoder_frozen") is not True:
            raise ValueError("Frozen image cache requires a frozen encoder")
        self.hidden_size = hidden_size
        self.paths = {}
        for row in metadata["records"]:
            digest = row["source_image_sha256"]
            path = Path(row["features"])
            if digest in self.paths and self.paths[digest] != path:
                raise ValueError("Ambiguous cached image preprocessing")
            self.paths[digest] = path

    def materialize(self, refs: MMRefs, sequence_length: int) -> dict[str, torch.Tensor]:
        mask = torch.zeros((1, sequence_length), dtype=torch.bool)
        features = []
        # masked_scatter consumes features in token order, independent of reference order.
        for ref in sorted(refs.images, key=lambda ref: ref.offset):
            if ref.offset < 0 or ref.length <= 0 or ref.offset + ref.length > sequence_length:
                raise ValueError("Image span outside packed sequence")
            span = mask[:, ref.offset : ref.offset + ref.length]
            if span.any():
                raise ValueError("Overlapping image spans")
            header, separator, payload = ref.url.partition(",")
            if not separator or not header.startswith("data:image/") or ";base64" not in header:
                raise ValueError("Frozen image lookup requires base64 image data")
            digest = hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest()
            if digest not in self.paths:
                raise ValueError("Image has no matching frozen encoder cache entry")
            tensor = torch.load(self.paths[digest], map_location="cpu", weights_only=True)["image_features"]
            if tensor.shape != (ref.length, self.hidden_size) or not torch.isfinite(tensor).all():
                raise ValueError("Cached image feature shape/count or finiteness mismatch")
            features.append(tensor)
            span.fill_(True)
        if not features:
            raise ValueError("No image references to materialize")
        return {"image_features": torch.cat(features), "image_token_mask": mask}
