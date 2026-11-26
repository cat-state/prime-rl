# Bounded RGB WGSL renderer

The model-facing interface is `fn paint(uv: vec2f) -> vec3f`; see `contract.txt` for all restrictions. UV origin is top-left, samples are pixel centers, and output is display RGB clamped to [0,1] then rounded to PNG8. There is no gamma conversion or target texture. Source helpers, scalar/vector math and conditionals are supported; loops, arrays, structs, module variables and entry points are rejected. Naga validates both user code and the composed compute shader. Expanded work is capped at 4096 nodes with helper nesting at most 16. Resolution is 1..512 per axis, default 256².

Each render runs a separate native process/device. It receives only the WGSL text, dimensions and trusted output filename. WGSL has no file/network API; user-declared bindings and entry points are rejected before adapter creation. The wrapper applies a 20-second wall deadline (configurable up to 30), 15-second CPU limit and 16-MiB file limit, kills the process group on timeout and retains source/error evidence. GPU readback has a five-second wait limit. The static loop/work restrictions reduce long-kernel risk; process termination alone is not a guarantee that a driver has canceled an already submitted command.

GPU ownership is explicit through `PI_GPU_PCI_BUS_ID`, matched against Vulkan adapter PCI metadata and checked again by the wrapper. `CUDA_VISIBLE_DEVICES` is not used as a Vulkan ownership mechanism. No fallback adapter is selected. All shaders execute without randomness; repeated renders on this backend should agree to PNG quantization, though cross-device floating-point differences should be assessed with meaningful image tolerances.

Build on the existing node, reusing its cargo cache without replacing the SDF binary:

```sh
cd /root/prime-rl/tools/painting_dataset/rgb_renderer
CARGO_TARGET_DIR=/root/prime-rl/loop-validation/painting-render-build cargo build --release --locked --offline
```

Only `painting-rgb-renderer` is produced; existing `render_server` and SDF sources stay intact. Set `PAINTING_RGB_RENDERER_BINARY` to override the binary path. Python uses stdlib plus the existing Pillow/NumPy for validation; no venv edits or new Python dependencies.

```python
from tools.painting_dataset.rgb_renderer import factory

with factory(workspace, pci_bus_id="0000:05:00.0") as renderer:
    result = renderer.render(source, turn_dir, width=256, height=256)
    # result: {ok, image, error, pci_bus_id, wall_seconds, artifact_dir, ...}
```

The context owns a temporary private working directory and cleans it when closed. Every call creates a unique artifact directory under the supplied turn directory, retaining source, PNG, stderr and result JSON. Factories share no native process, pipeline or mutable output path. Use separate contexts for separate rollouts.

Validation from the project root:

```sh
PI_GPU_PCI_BUS_ID=0000:05:00.0 uv run --no-sync python -m tools.painting_dataset.rgb_renderer.validate loop-validation/rgb-renderer-validation
```

Validated on RTX PRO 6000 Blackwell: constant output error 0, analytic x/y gradient maximum error 0.00194547 versus half-byte quantization tolerance 0.00196178. Invalid syntax, loops, external bindings, oversized source and wrong GPU fail; forced wall timeout kills the child, subsequent rendering succeeds, and private directories are cleaned. Timeout validation forces a deadline before dispatch rather than submitting an unbounded GPU kernel. The existing SDF renderer is not involved.
