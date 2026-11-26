"""Private process/socket ownership for the existing WGSL SDF renderer."""
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from tools.painting_dataset.agentic_episode import confined


class PrivateSDFRenderer:
    def __init__(self, workspace: Path, *, executable: Path, source: Path,
                 gpu_pci_bus_id: str, timeout: float = 90):
        self.workspace = workspace.resolve()
        self.executable, self.source = executable.resolve(), source.resolve()
        self.gpu_pci_bus_id, self.timeout = gpu_pci_bus_id, timeout
        self.socket = self.workspace / "render.sock"
        if len(os.fsencode(self.socket)) >= 108:
            raise ValueError("Episode workspace path too long for private UNIX socket")
        self.process = None
        self.log = None

    def __enter__(self):
        self.start()
        return self

    def start(self):
        self.socket.unlink(missing_ok=True)
        self.log = (self.workspace / "renderer.log").open("a")
        env = dict(os.environ, PI_GPU_PCI_BUS_ID=self.gpu_pci_bus_id, WGPU_BACKEND="vulkan",
                   PI_RENDER_BUDGET_MS=str(int(self.timeout * 900)))
        self.process = subprocess.Popen([str(self.executable), "--src", str(self.source), "--socket", str(self.socket)],
                                        cwd=self.workspace, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        deadline = time.monotonic() + 30
        while not self.socket.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise RuntimeError("Private renderer failed to start; see renderer.log")
            time.sleep(.05)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
            self.log = None
        self.socket.unlink(missing_ok=True)

    def __exit__(self, *args):
        self.close()

    def render(self, source: str, directory: Path, width: int, height: int) -> dict:
        directory = confined(directory, self.workspace)
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.start()
        request = {"scene": 9, "seed": 42, "width": width, "height": height, "samples": 1,
                   "eye": [0, 0, 4], "look": [0, 0, 0], "name": "render", "out": str(directory),
                   "shader_src": source}
        try:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(self.timeout)
                client.connect(str(self.socket))
                client.sendall((json.dumps(request) + "\n").encode())
                response = json.loads(client.makefile("rb").readline(2_000_000))
        except (TimeoutError, OSError, ValueError):
            self.close()
            raise
        for path in response.get("files", []):
            confined(path, directory)
        if response.get("ok"):
            response["image"] = str(confined(directory / "render-seed-42.png", directory))
        return response
