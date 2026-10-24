"""Validate every A/B boundary against upstream NCM generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ncm_serving.engine import NCMLevelExecutor
from ncm_serving.runtime import load_wmt14_runtime


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncm-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    started = time.monotonic()
    runtime = load_wmt14_runtime(
        args.ncm_root, args.checkpoint_dir, args.tokenizer_dir
    )
    prefixes = runtime.encode(args.prompt)
    executor = NCMLevelExecutor(runtime, split=5)
    rows = []
    for cfg in (0.0, 1.0):
        for split in range(1, len(executor.levels)):
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            with torch.no_grad():
                expected_codes = runtime.net.generate(
                    None,
                    prefixes.to(executor.device),
                    temperature=float(runtime.cfg.generation.temperature),
                    top_k=int(runtime.cfg.generation.top_k),
                    top_p=float(runtime.cfg.generation.top_p),
                    cfg_scale=cfg,
                    cfg_null_eos=runtime.eos,
                )
                _, levels = runtime.tokenizer.vqvae.vq.idx_to_zhats(
                    expected_codes, return_zlevels=True
                )
                expected_ids = runtime.tokenizer.vqvae.decoder(levels[-1]).argmax(-1)
            executor.split = split
            state = executor.submit(
                f"cfg{cfg:g}-split{split}", prefixes, seed=args.seed, cfg=cfg
            )
            executor.short_group(state)
            actual = executor.long_group(executor.handoff(state))
            if not torch.equal(actual["codes"], expected_codes):
                raise AssertionError(f"code mismatch for cfg={cfg}, split={split}")
            if not torch.equal(actual["token_ids"], expected_ids):
                raise AssertionError(f"token ID mismatch for cfg={cfg}, split={split}")
            rows.append(
                {
                    "cfg": cfg,
                    "split": split,
                    "samples": len(args.prompt),
                    "codes_sha256": tensor_hash(actual["codes"]),
                    "token_ids_sha256": tensor_hash(actual["token_ids"]),
                    "status": "PASS",
                }
            )
    payload = {
        "status": "PASS",
        "kind": "correctness-only",
        "cells": rows,
        "elapsed_wall_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(executor.device),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(rows)} boundary cells -> {args.output}")


if __name__ == "__main__":
    main()
