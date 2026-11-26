"""Private-process RGB WGSL rendering with bounded source, dimensions and runtime."""

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


class RGBRenderer:
    def __init__(self, workspace, *, binary=None, pci_bus_id=None, timeout=20.0):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        default = (
            Path(__file__).resolve().parents[3] / "loop-validation/painting-render-build/release/painting-rgb-renderer"
        )
        self.binary = Path(binary or os.environ.get("PAINTING_RGB_RENDERER_BINARY", default)).resolve()
        self.pci = pci_bus_id or os.environ.get("PI_GPU_PCI_BUS_ID")
        if not self.pci:
            raise ValueError("Set PI_GPU_PCI_BUS_ID to the reserved Vulkan GPU PCI bus ID")
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        self.timeout = float(timeout)
        if not 0 < self.timeout <= 30:
            raise ValueError("timeout must be in (0, 30] seconds")
        self._private = None

    def __enter__(self):
        self._private = tempfile.TemporaryDirectory(prefix="rgb-renderer-", dir=self.workspace)
        return self

    def __exit__(self, *_):
        if self._private:
            self._private.cleanup()
            self._private = None

    def render(self, source, turn_dir, width=256, height=256):
        if self._private is None:
            raise RuntimeError("Use the renderer as a context manager")
        if not isinstance(source, str) or len(source.encode()) > 32768:
            return {"ok": False, "image": None, "error": "Source exceeds 32768 UTF-8 bytes or is not text"}
        if type(width) is not int or type(height) is not int or not (1 <= width <= 512 and 1 <= height <= 512):
            return {"ok": False, "image": None, "error": "Dimensions must be integers in 1..512"}
        destination = Path(turn_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="rgb-", dir=destination))
        (output_dir / "source.wgsl").write_text(source)
        request = {"source": source, "width": width, "height": height, "output": str(output_dir / "image.png")}
        env = dict(os.environ, PI_GPU_PCI_BUS_ID=self.pci, WGPU_BACKEND="vulkan")
        started = time.monotonic()
        with subprocess.Popen(
            ["/usr/bin/prlimit", "--cpu=15:16", "--fsize=16777216", "--core=0", str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._private.name,
            env=env,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(json.dumps(request), timeout=self.timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                result = {"ok": False, "image": None, "error": "Renderer wall-time budget exceeded", "timed_out": True}
            else:
                try:
                    result = json.loads(stdout)
                except json.JSONDecodeError:
                    result = {"ok": False, "image": None, "error": "Renderer exited without valid JSON"}
                if process.returncode != 0 or not result.get("ok"):
                    result.update(ok=False, image=None)
                elif result.get("pci_bus_id", "").lower() != self.pci.lower():
                    result = {"ok": False, "image": None, "error": "GPU ownership mismatch"}
                elif not Path(request["output"]).is_file():
                    result = {"ok": False, "image": None, "error": "Renderer did not create the PNG"}
            result.update(
                returncode=process.returncode, wall_seconds=time.monotonic() - started, artifact_dir=str(output_dir)
            )
        (output_dir / "stderr.txt").write_text(stderr)
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        return result


def factory(workspace, **kwargs):
    return RGBRenderer(workspace, **kwargs)
