"""Run one request through the single-device serving core."""

from __future__ import annotations

import argparse
import json

from .runtime import load_wmt14_runtime
from .service import NCMServingSystem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncm-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", type=int, default=5)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--cfg", type=float, choices=(0.0, 1.0), default=0.0)
    args = parser.parse_args()
    runtime = load_wmt14_runtime(
        args.ncm_root,
        args.checkpoint_dir,
        args.tokenizer_dir,
        device=args.device,
    )
    serving = NCMServingSystem(runtime, split=args.split, prefix_cache=args.prefix_cache)
    result = serving.generate_text(args.prompt, seed=args.seed, cfg=args.cfg)
    print(json.dumps({"texts": result["texts"], "token_ids": result["token_ids"].tolist()}))


if __name__ == "__main__":
    main()
