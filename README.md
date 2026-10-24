# NCM Serving

An independent serving runtime for grouping Next-Concept Model generation into
short and long concept-level stages.

This repository contains the serving layer only. It does not vendor the
[HazyResearch NCM](https://github.com/HazyResearch/ncm) model implementation,
weights, tokenizer, or datasets. The current adapter targets NCM commit
`c70274273f489abc73832b2b98584a8483b6b203`.

## What is implemented

- Exact request pause/resume at any concept-level boundary.
- Logical A/B execution groups with a configurable split.
- Per-request codes, latent state, masks, model identity, and CPU/CUDA RNG.
- Ownership-safe handoff, stale-state rejection, cancellation, and request queues.
- A synchronous serving API and CLI for the WMT14 raw-prefix NCM path.

The validated default is `split=5`: group A executes concept lengths
`[1,4,8,14,20]`; group B executes `[26,32,40,48,56,64]`. Both groups currently
run sequentially on one GPU and share one model instance. See
[design](docs/design.md) and [validation](docs/validation.md) for exact scope.

## Install

Install a CUDA-compatible PyTorch build first, then install this package:

```bash
python -m pip install -e .
```

Keep a separate NCM checkout at the pinned commit and download its WMT14
checkpoint plus Qwen tokenizer files according to the upstream instructions.

## Python API

```python
from ncm_serving import NCMServingSystem
from ncm_serving.runtime import load_wmt14_runtime

runtime = load_wmt14_runtime(
    ncm_root="../ncm",
    checkpoint_dir="../ncm/artifacts/wmt14",
    tokenizer_dir="../ncm/artifacts/qwen3-0.6b-tokenizer",
)
server = NCMServingSystem(runtime, split=5)
result = server.generate_text(
    ["Ein Beispieltext"],
    seed=20261023,
    cfg=1.0,
)
print(result["texts"])
```

Requests may also be submitted separately and advanced with
`server.scheduler.step()`, allowing multiple requests to wait in the A/B queues.

## CLI

```bash
ncm-serving \
  --ncm-root ../ncm \
  --checkpoint-dir ../ncm/artifacts/wmt14 \
  --tokenizer-dir ../ncm/artifacts/qwen3-0.6b-tokenizer \
  --prompt "Ein Beispieltext" \
  --cfg 1
```

## Test

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The GPU boundary validator in `scripts/validate_wmt14.py` compares this serving
executor with the upstream `net.generate` path. It is a correctness check, not a
performance benchmark.

The current release is a correctness-validated single-device serving core. It
does not yet implement real multi-GPU transport, cross-request dynamic batching,
or performance claims.
