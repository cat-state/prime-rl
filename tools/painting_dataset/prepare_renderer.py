"""Prepare an isolated centered-camera shader tree, preserving the campaign source."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    args.destination.mkdir(parents=True, exist_ok=False)
    original = (source / "spaces.wgsl").read_text()
    offset = "if u.config.z==8u {uv+=vec2f(.60*f32(u.config.x)/f32(u.config.y),-.08);}"
    if original.count(offset) != 1:
        raise ValueError("Expected exactly one known Crivelli camera offset")
    centered = original.replace(offset, "")
    for path in source.iterdir():
        if path.name != "spaces.wgsl":
            (args.destination / path.name).symlink_to(path, target_is_directory=path.is_dir())
    (args.destination / "spaces.wgsl").write_text(centered)
    (args.destination / "camera-variant.json").write_text(json.dumps({
        "source": str(source), "variant": "centered perspective, vertical FOV 48 degrees",
        "change": "Remove scene9-specific projection offset only; lighting/materials unchanged",
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "centered_sha256": hashlib.sha256(centered.encode()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
