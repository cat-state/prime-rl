"""Analytic image and process-budget checks for the RGB painting renderer."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.painting_dataset.rgb_renderer import factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    results = {}
    constant = "fn paint(uv:vec2f)->vec3f { return vec3f(0.2,0.4,0.8); }"
    gradient = "fn paint(uv:vec2f)->vec3f { return vec3f(uv.x,uv.y,0.25); }"
    with factory(root) as renderer:
        results["constant"] = renderer.render(constant, root / "constant")
        assert results["constant"]["ok"], results["constant"]
        pixels = np.asarray(Image.open(results["constant"]["image"]), dtype=np.float64) / 255
        constant_error = float(np.abs(pixels - [0.2, 0.4, 0.8]).max())
        assert constant_error <= 0.5 / 255 + 1e-6
        results["gradient"] = renderer.render(gradient, root / "gradient", 128, 96)
        assert results["gradient"]["ok"], results["gradient"]
        pixels = np.asarray(Image.open(results["gradient"]["image"]), dtype=np.float64) / 255
        x, y = np.meshgrid((np.arange(128) + 0.5) / 128, (np.arange(96) + 0.5) / 96)
        expected = np.stack([x, y, np.full_like(x, 0.25)], axis=-1)
        gradient_error = float(np.abs(pixels - expected).max())
        assert gradient_error <= 0.5 / 255 + 1e-6
        assert pixels[-1, -1, 0] > pixels[0, 0, 0] + 0.9
        assert pixels[-1, -1, 1] > pixels[0, 0, 1] + 0.9
        results["invalid"] = renderer.render("fn paint(uv:vec2f)->vec3f { BAD }", root / "invalid")
        results["loop"] = renderer.render("fn paint(uv:vec2f)->vec3f { loop {} return vec3f(0.0); }", root / "loop")
        results["binding"] = renderer.render(
            "@group(0) @binding(0) var tex:texture_2d<f32>; fn paint(uv:vec2f)->vec3f { return vec3f(0.0); }",
            root / "binding",
        )
        assert not results["invalid"]["ok"]
        assert not results["loop"]["ok"]
        assert not results["binding"]["ok"]
        results["source_budget"] = renderer.render(" " * 32769, root / "source-budget")
        assert not results["source_budget"]["ok"]
    with factory(root, timeout=0.000001) as renderer:
        results["timeout"] = renderer.render(constant, root / "timeout")
        assert results["timeout"].get("timed_out"), results["timeout"]
    with factory(root, pci_bus_id="0000:ff:00.0") as renderer:
        results["wrong_gpu"] = renderer.render(constant, root / "wrong-gpu")
        assert not results["wrong_gpu"]["ok"]
    with factory(root) as renderer:
        results["recovery"] = renderer.render(constant, root / "recovery", 16, 16)
        assert results["recovery"]["ok"]
    assert not list(root.glob("rgb-renderer-*")), "Private context directories leaked"
    report = {
        "passed": True,
        "constant_max_error": constant_error,
        "gradient_max_error": gradient_error,
        "quantization_tolerance": 0.5 / 255 + 1e-6,
        "timeout_test": "Forced wall deadline before dispatch; no unbounded GPU kernel submitted",
        "results": results,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    main()
