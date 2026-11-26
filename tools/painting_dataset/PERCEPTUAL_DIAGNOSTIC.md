# Pinned spatial perceptual diagnostic

`perceptual_reward.py` compares aligned DINOv2-small patch embeddings, averaging cosine similarity equally over layers 3, 6, 9 and 12 and over corresponding image patches. Global CLS similarity and last-layer-only patch similarity are recorded as ablations, not used in the primary score. No model weights or score coefficients are trained. The function is diagnostic and is not automatically wired into RL.

Model: `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, safetensors SHA256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` (88,249,960 bytes). Existing Transformers/PyTorch dependencies suffice; no new package or venv modification is needed.

Primary sources: [DINOv2 authors' repository](https://github.com/facebookresearch/dinov2), [paper](https://arxiv.org/abs/2304.07193), [official-account model weights/card](https://huggingface.co/facebook/dinov2-small). The HF card notes that it was written by Hugging Face, not the original paper authors. These sources establish pretrained feature extraction, not the validity of this custom painting metric. [LPIPS authors' implementation](https://github.com/richzhang/PerceptualSimilarity) was considered; LPIPS was not installed or tested in this experiment.

Input controls are rendered at at most 256 pixels per axis, then aspect-preserving bicubic thumbnails at at most 224, with constant padding and ImageNet normalization. Valid-area fractions weight the 14-pixel patch grid; no center crop removes composition. Attention still sees padding, so aspect/letterbox effects remain a limitation. Inputs smaller than 224 are not upsampled; use the same rendering geometry and scale for both sides. Score corresponding positions, not a best-match bag of patches.

```sh
CUDA_VISIBLE_DEVICES=2 uv run --no-sync python -m tools.painting_dataset.perceptual_reward \
  loop-validation/painting-data-nga-v1/manifest.json \
  loop-validation/perceptual-reward-calibration-v1 \
  --episodes loop-validation/agentic-real-art8192-v1
```

Only use an explicitly reserved GPU. The measured diagnostic footprint was about 1 GiB on the existing GPU node. `--limit 0` skips museum controls and scores supplied completed episode directories only; create a distinct output directory to retain evidence. Do not run this against partially written episodes.

Initial calibration: all 100 museum targets pass blur ordering, mosaic-over-wash, small-over-large translation, and blur-over-flipped/wrong artwork comparisons; 97/100 pass mosaic16 over shuffled blocks. Global CLS is markedly worse at spatial controls. Switching to the last layer because it looks better on one artwork degrades the 100-image block-shuffle comparison from 97/100 to 60/100; retain the predeclared average.

For the initial Wheat Field with Cypresses reconstruction, spatial scores put both recognizable shaders (0.3927 / 0.3867) above mean/gradient washes (0.3004 / 0.3158), correcting the raw color/edge metric's ordering. However, shuffled pixels score 0.3768, close to the crude shaders. Three control failures favor detailed scrambled fragments over coarse but coherent mosaics. These are real limitations: pretrained similarity is not a guarantee of composition or artistic quality. Additional generated candidates and held-out visual review are required before any bounded optimization pilot, with raw components and controls retained.
