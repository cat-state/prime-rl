"""Prepare blind visual prompts and collect resumable Inkling scene descriptions."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Describe this rendered image in 3–5 concise sentences. Identify the visible objects, geometry, spatial layout, colors, materials, lighting, and viewpoint. Describe only what is visible; do not infer a title or unseen objects.",
    "Inspect this image as a 3D scene artist. Give five short labeled sections: Geometry; Layout and camera; Materials and colors; Lighting; Uncertainties. Be specific about visible shapes and relationships. Separate observations from guesses, and do not invent hidden structure.",
    "Describe this image carefully for someone who cannot see it. First list three unmistakable visual facts, then give a short overall description. Finally identify any ambiguous feature and two plausible interpretations, if there is one. Do not force an interpretation or guess a scene name.",
]


def post(endpoint, payload):
    request = urllib.request.Request(f"http://127.0.0.1:8766/{endpoint}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "describe"])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.dataset / "manifest.json").read_text())["records"]
    if args.mode == "prepare":
        args.output.mkdir(exist_ok=False)
        prepared = []
        # Direct descriptions cover the catalog; elicitation variants cover its first nine scenes.
        for round_index, prompt in enumerate(PROMPTS):
            for row in rows:
                if round_index and row["scene"] > 9:
                    continue
                prepared.append({**row, "id": f"{row['id']}-round{round_index}",
                                 "prompt": prompt, "round": round_index})
        (args.output / "manifest.json").write_text(json.dumps({"records": prepared}, indent=2))
        return
    args.output.mkdir(exist_ok=True)
    required = {r["id"] for r in rows}
    for _ in range(150):
        try:
            health = json.load(urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=2))
            if required.issubset(health["prompts"]):
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Description prompt cache unavailable")
    policy = "counting-group-step1"
    if policy not in health["policies"]:
        post("adapters", {"policy": policy, "path": "/root/prime-rl/loop-validation/counting-hard-task/adapter-group-step1"})
    for row in rows:
        for name in ["base", policy]:
            path = args.output / f"{row['id']}-{name}.json"
            payload = dict(prompt_id=row["id"], policy=name, max_tokens=2048, temperature=0.3, seed=42)
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["request"] != payload or saved["image_sha256"] != row["image_sha256"] or saved["prompt"] != row["prompt"]:
                    raise ValueError(f"Prior evidence differs: {path}")
                continue
            response = post("generate", payload)
            text = response["raw_output"]
            final = text.rsplit("<|content_text|>", 1)[1].split("<|end_message|>", 1)[0].strip() if "<|content_text|>" in text else None
            record = dict(request=payload, response=response, prompt=row["prompt"], image_sha256=row["image_sha256"],
                          scene_name=row["scene_name"], image=row["image"], final=final)
            path.write_text(json.dumps(record))
            print(dict(id=row["id"], policy=name, finish=response["finish_reason"],
                       tokens=len(response["completion_ids"]), final=final), flush=True)


if __name__ == "__main__":
    main()
