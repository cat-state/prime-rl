import base64
import hashlib
import json

import pytest
import torch

from prime_rl.trainer.models.inkling2_loop.frozen_images import FrozenImageCache
from prime_rl.trainer.rl.data import prepare_micro_batch
from prime_rl.transports.batch.types import MMImageRef, MMRefs


def test_packed_image_order_and_span_validation(tmp_path):
    records = []
    urls = []
    for index in (1, 2):
        content = bytes([index])
        urls.append("data:image/png;base64," + base64.b64encode(content).decode())
        path = tmp_path / f"{index}.pt"
        torch.save({"image_features": torch.full((2, 3), float(index))}, path)
        records.append({"source_image_sha256": hashlib.sha256(content).hexdigest(), "features": str(path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"encoder_frozen": True, "records": records}))
    cache = FrozenImageCache(manifest, hidden_size=3)
    refs = MMRefs(images=[MMImageRef(urls[1], 6, 2), MMImageRef(urls[0], 1, 2)])
    result = cache.materialize(refs, sequence_length=10)
    assert result["image_token_mask"].nonzero()[:, 1].tolist() == [1, 2, 6, 7]
    torch.testing.assert_close(result["image_features"][:, 0], torch.tensor([1., 1., 2., 2.]))
    batch = {"input_ids": torch.zeros((1, 10), dtype=torch.long), "mm_refs": refs,
             "mm_token_type_ids": result["image_token_mask"].long()}
    prepared = prepare_micro_batch(batch, None, cache.materialize)
    assert prepared["mm_refs"] is None
    torch.testing.assert_close(prepared["mm_kwargs"]["image_features"], result["image_features"])
    batch["mm_refs"] = refs
    batch["mm_token_type_ids"] = torch.zeros((1, 10), dtype=torch.long)
    with pytest.raises(ValueError, match="rollout image token metadata"):
        prepare_micro_batch(batch, None, cache.materialize)
    for invalid in (
        [MMImageRef(urls[0], 9, 2)],
        [MMImageRef(urls[0], 1, 2), MMImageRef(urls[1], 2, 2)],
        [MMImageRef(urls[0], 1, 3)],
    ):
        with pytest.raises(ValueError):
            cache.materialize(MMRefs(images=invalid), sequence_length=10)
