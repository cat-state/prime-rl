"""Export the released Mini text weights as BF16 for PrimeRL's inference worker.

The output needs about 52 GB. The original checkpoint remains unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.inkling2_loop import Inkling2LoopConfig
from prime_rl.trainer.models.inkling2_loop.converting_inkling2_loop import conversion_chain
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.weights import load_state_dict, save_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    setup_logger("info")
    source = Path(args.source)
    if not source.is_dir():
        source = Path(snapshot_download(args.source, revision=args.revision))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be a new or empty directory")
    config = Inkling2LoopConfig.from_dict(json.loads((source / "config.json").read_text()))
    weights = load_state_dict(source)
    ops = conversion_chain(config)
    apply_hf_to_prime(weights, ops)
    apply_prime_to_hf(weights, ops)
    args.output.mkdir(parents=True, exist_ok=True)
    save_state_dict(weights, args.output)
    (args.output / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    for pattern in ("tokenizer*", "special_tokens_map.json", "generation_config.json", "chat_template*", "*.tiktoken"):
        for path in source.glob(pattern):
            if path.is_file():
                shutil.copyfile(path, args.output / path.name)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
