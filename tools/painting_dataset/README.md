# Painting reconstruction seed dataset

Run from `/root/prime-rl`:

```sh
uv run --no-sync python tools/painting_dataset/build.py /root/prime-rl/loop-validation/painting-data
```

The builder fetches six paintings from the Met's public API, requires its public-domain flag, validates image decoding, records source metadata and content hashes, and writes four training and two held-out artwork records. Cached images/metadata are reused. Failures propagate; a failed build does not claim a complete manifest. Images are museum web-large derivatives with original aspect ratios. No crops or augmented versions may cross the artwork split. This tiny seed set supports integration, not statistically credible generalization claims. `target_code` is null: these are RL targets, not invented supervised shader labels.

## Existing renderer and Crivelli source

Use the preserved campaign renderer at `/root/simple_pt/research/w768_8192_wgsl_renderonly_1h_20261019/renderer` as the integration baseline. Its `launch-pins.json` records historical shader hashes; sibling campaign copies exist and are not newly duplicated here. `src/crivelli.wgsl` and `src/crivelli_figures.wgsl` implement Carlo Crivelli's *The Annunciation with Saint Emidius* procedurally. No original Crivelli reference image was found in the local tracked inventory or the remote filename search. These shaders are an implementation example, **not** a ground-truth photograph of the painting. Add a properly sourced reference before making a Crivelli image task. Keep any Crivelli-derived targets in training because its code is available as an example.

The source-inspected server invocation is:

```sh
WGPU_BACKEND=vulkan PI_RENDER_BUDGET_MS=60000 /root/simple_pt/research/w768_8192_wgsl_renderonly_1h_20261019/renderer/target/release/render_server --src /root/simple_pt/research/w768_8192_wgsl_renderonly_1h_20261019/renderer/src --socket /tmp/painting-render.sock
```

The binary/runtime availability and actual rendering still require integration validation. Existing campaign launchers sometimes supply a Vulkan `LD_LIBRARY_PATH`; reuse their runtime only after checking dependencies. No GPU work is launched by the dataset builder.

The server accepts newline-delimited JSON over a Unix socket. Scene IDs are one-based. A baseline Crivelli request is:

```json
{"scene":9,"seed":42,"width":384,"height":512,"samples":1,"eye":[-0.7,5.9,14],"look":[-0.7,5.9,0],"name":"crivelli-baseline","out":"/root/prime-rl/loop-validation/painting-data/renders"}
```

For a candidate, add `"shader_src":"...WGSL..."` defining `fn scene_override(p: vec3f) -> vec2f`. Return signed distance and an existing palette material ID. Helpers are assembled from the renderer's source; supply their documented signatures and palette to the model. The server replaces the environment dispatch, not the entire shader. Arbitrary fragment shaders are **not** compatible. This geometry/fixed-palette contract is restrictive for arbitrary paintings; a separately validated 2D color shader renderer may be better later. Do not score arbitrary text as if it rendered.

## Environment integration

- Input: one image from `image`, plus the prompt and renderer helper/palette contract. Preserve aspect ratio; record exact resize/letterbox transform. Art metadata is provenance, not an answer.
- Action: WGSL source only. Strip an optional single code fence, enforce source-length and rendering-time budgets, compile with Naga, then render in an isolated process. Reject external resource access and pipeline/interface declarations. The existing server accepts general WGSL text; containment must be provided by the environment rather than assumed.
- Reward proposal: invalid compilation, timeout, or nonfinite pixels receive zero. For valid RGB output, compare identically resized image interiors using multiscale color/edge error, with a perceptual similarity term only after validating a pinned implementation. Report components separately. Do not grant a large compile-only reward that can dominate visual quality. Calibrate with constant-color, shuffled-target and existing procedural baselines before training.
- Evaluation: hold out complete artworks and all their crops. Report target/candidate thumbnails, compile rate, render latency, reward components, and comparison to the initial model; never treat improvement on four training images as broad artistic ability.

Pending: Crivelli reference acquisition, rendered shader sanity checks, exact palette/helper prompt, environment wrapper, calibrated reward and image RL validation. Existing gsplat assets are untouched.
