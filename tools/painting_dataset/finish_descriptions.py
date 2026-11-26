"""Finish the catalog description queue after its preview and rendering stages."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    root = Path("/root/prime-rl")
    os.chdir(root)
    data = root / "loop-validation/wgsl-descriptions"
    preview = root / "loop-validation/wgsl-description-preview"
    full = root / "loop-validation/wgsl-description-full"
    uv = "/root/.local/bin/uv"
    expected = 2 * len(json.loads((preview / "manifest.json").read_text())["records"])
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        if (data / "render-complete.json").exists() and len(list((data / "captions").glob("*.json"))) >= expected:
            break
        time.sleep(15)
    else:
        raise TimeoutError("Rendering/preview stage did not finish within six hours")
    subprocess.run([uv, "run", "--no-sync", "python", "-m", "tools.painting_dataset.describe_environments",
                    "prepare", str(data), str(full)], check=True)
    subprocess.run([uv, "run", "--project", "/root/inkling-mini", "--no-sync", "python",
                    "tools/painting_dataset/encode.py", str(full)], env={**os.environ, "CUDA_VISIBLE_DEVICES": "3"}, check=True)
    # Stop only the preview server, after every preview request has returned and been saved.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError):
            continue
        if b"tools/painting_dataset/serve.py" not in command or b"loop-validation/wgsl-description-preview/features/manifest.json" not in command:
            continue
        if not Path(os.fsdecode(command[0])).name.startswith("python"):
            continue
        pid = int(proc.name)
        children = (proc / "task" / proc.name / "children").read_text().split()
        for child in children:
            child_path = Path("/proc") / child / "cmdline"
            if child_path.exists() and b"VLLM::EngineCore" in child_path.read_bytes():
                os.kill(int(child), signal.SIGTERM)
        os.kill(pid, signal.SIGTERM)
    time.sleep(10)
    with (root / "wgsl-description-full-server.log").open("x") as log:
        server = subprocess.Popen([uv, "run", "--no-sync", "python", "tools/painting_dataset/serve.py",
                                   str(full / "features/manifest.json")],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": "1,2", "INKLING_MULTIMEM_AR": "0"},
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print({"full_server_pid": server.pid}, flush=True)
    subprocess.run([uv, "run", "--no-sync", "python", "-m", "tools.painting_dataset.describe_environments",
                    "describe", str(full), str(data / "captions")], check=True)
    (data / "descriptions-complete.json").write_text(json.dumps({"records": len(list((data / "captions").glob("*.json")))}))
    print("Catalog description queue complete", flush=True)


if __name__ == "__main__":
    main()
