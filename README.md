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

## Validation and benchmark history

All results below use upstream NCM commit
`c70274273f489abc73832b2b98584a8483b6b203`. The full WMT14 runs used Python
3.11.15, PyTorch 2.6.0+cu118, one RTX 4060 8 GiB (driver 560.94), FP32 with
TF32 disabled, seed 20261023, temperature 1, top-k 1, and top-p 0.

### Correctness and quality results

| Evaluation | Workload | Result | Status |
| --- | --- | --- | --- |
| Package lifecycle tests | Queue priority, handoff, cancellation, duplicate IDs, invalid checkout | 5/5 tests passed | Valid |
| Migrated executor CPU reference | 2 split points × CFG 0/1 × greedy/stochastic sampling | 8/8 cells; codes and IDs exact, latent within `atol=1e-4`, `rtol=1e-3`, external RNG unchanged | Valid |
| Packaged executor GPU boundary sweep | 10 split points × CFG 0/1 × 4 fresh prompts | 20/20 cells; codes and final IDs exact | Valid |
| Full WMT14 reference parity | `split=5`, batch 4, CFG 0/1, 3003 samples each | 6006/6006 samples; all 313 concept codes, 64 final IDs, and decoded text exact | Valid |
| Probe transparency | Batch 1/4/8 × CFG 0/1 | 6/6 cells exact; all 11 Transformer-stack calls observed | Valid correctness check, not a speed benchmark |

The full WMT14 outputs have the following task metrics. These scores describe
the shared exact output from upstream NCM, the original validation harness, and
this grouped execution path; they are not an acceleration result.

| CFG | Samples | BLEU | Lowercase BLEU | chrF | Length ratio (13a) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3003 | 21.720218 | 22.612115 | 49.564860 | 0.930454 |
| 1 | 3003 | 22.946409 | 23.835817 | 51.515977 | 0.948186 |

Exact output hashes and additional methodology details are recorded in
[the validation report](docs/validation.md).

### Historical performance runs

An earlier F00 run produced 60 end-to-end windows: upstream and harness paths,
batch sizes 1/4/8, CFG 0/1, five windows per cell, and 1024 completed samples
per window. The observed median throughput was:

| Batch | CFG | Upstream req/s | Harness req/s | Acceptance status |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 9.743 | 9.640 | Invalid for performance acceptance |
| 1 | 1 | 9.189 | 8.882 | Invalid for performance acceptance |
| 4 | 0 | 22.159 | 21.510 | Invalid for performance acceptance |
| 4 | 1 | 11.645 | 11.413 | Invalid for performance acceptance |
| 8 | 0 | 23.650 | 22.340 | Invalid for performance acceptance |
| 8 | 1 | 11.835 | 11.842 | Invalid for performance acceptance |

These numbers are retained for audit only. They cannot be used to claim a
speedup or regression because the harness left autograd enabled during latent
reconstruction and decoding, source files changed between worker processes,
upstream and harness windows were not interleaved, and the original manifest
did not freeze source hashes.

The earlier phase profile contained 30 windows (batch 1/4/8 × CFG 0/1 × five
windows), with 16 batches per window and 11 Transformer calls per batch. It is
also diagnostic-only: it used synchronization-heavy wall timing, omitted the
complete level-0 stack from attribution, and did not separate GPU event time
from CPU enqueue time. The instrumentation has since been corrected and passed
the six-cell transparency check above, but the corrected performance matrix has
not yet been run. Therefore this project currently makes **no performance
claim**.

The untracked local evidence contains 119 files (about 111 MiB), including full
codes, IDs, decoded outputs, per-window timing records, phase records, manifests,
and resource accounting. Large raw outputs, checkpoints, datasets, and virtual
environments are intentionally excluded from Git.

## Roadmap

The following items are planned work, not implemented features or measured
benefits. Each optimization must preserve the validated output contract and be
evaluated separately before combinations are reported.

| Stage | Planned feature | Required validation |
| --- | --- | --- |
| R1 | Corrected single-GPU baseline and phase cost model | Frozen source manifest, no-grad parity, CUDA-event GPU time, separate CPU enqueue time, peak allocated/reserved memory |
| R2 | Real A/B workers on separate GPUs with serialized handoff state | Exact codes/IDs/text after transfer, model/config identity, RNG continuity, cancellation and delayed/duplicate message handling |
| R3 | Level-aware continuous batching | Request isolation under insert/finish/cancel/reorder, batch-shape quality parity, queue fairness and capacity limits |
| R4 | Dynamic split and A:B resource allocation | Cost model trained on validation traces, split selected without using the final test trace, equal-resource comparison |
| R5 | CUDA Graphs and launch-overhead reduction for short levels | Graph-safe buffers and RNG, unsupported-shape fallback, end-to-end gain beyond measurement noise |
| R6 | Long-level attention and matrix-compute optimization | Independent numerical oracle, real model shapes/dtypes, kernel and end-to-end ablations |
| R7 | Prompt-state reuse and cache placement | Complete cache key, reference counts, eviction, true workload hit rate, capacity and tail-latency impact |
| R8 | Production lifecycle controls | Deadlines, admission control, bounded queues, failure recovery, observability, and leak tests |

### Planned benchmark suite

| ID | Benchmark | Baseline and workload | Primary outputs |
| --- | --- | --- | --- |
| B01 | Corrected phase profile | Direct NCM and serving path; batch 1/4/8, CFG 0/1; five source-frozen windows | GPU and CPU time per phase/level, forward counts, peak memory, dispersion |
| B02 | Kernel microbenchmarks | Real short/long-level shapes, supported dtypes/layouts, eager PyTorch reference | Latency, numerical error, temporary memory, unsupported-shape fallback |
| B03 | Single-GPU end to end | At least 1024 WMT14 requests per window, five interleaved windows after stable warmup | Batch latency, requests/s, actual tokens/s, memory, failures; loading reported separately |
| B04 | Online serving load | Fixed open-loop traces at 0.25/0.5/0.75/1.0/1.25× measured saturation plus a burst; at least 10,000 requests for tail claims | Time-to-final-output p50/p95/p99, completed throughput, SLO goodput, rejection and timeout rates |
| B05 | Cache workloads | No reuse, natural repetition, and controlled sharing ratios under the same arrivals | Hit rate, cache bytes, recompute avoided, throughput and tail latency |
| B06 | Handoff and transfer | Actual state payloads for every split over supported PCIe/P2P links | Payload bytes, submit latency, effective bandwidth, synchronization and reconstruction cost |
| B07 | Migration policy | Stay local vs transfer latent vs transfer codes and rebuild | Completion time, transfer bytes, rebuild time, effect on concurrent requests |
| B08 | Transfer overlap and lossless encoding | Synchronous transfer vs overlap vs batching; uncompressed payload as baseline | End-to-end gain, encode/decode cost, overlap achieved, compute interference |
| B09 | Multi-GPU stage separation | Same GPU count: full-instance data parallel vs fixed A/B groups vs dynamic placement | Goodput/GPU, p50/p95/p99, utilization, memory, transfer pressure, failures |
| B10 | Ablations | One mechanism at a time, then the approved combination | Incremental benefit, negative results, interactions, and confidence/dispersion |

For every performance run, model/data revisions, input order, source hashes,
precision, sampling parameters, hardware, driver, attention backend, and raw
windows will be recorded. Quality parity is a prerequisite; CPU simulations do
not count as multi-GPU evidence, and equal-resource baselines are mandatory.

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

The current release is a correctness-validated single-device serving core. The
roadmap above does not imply that multi-GPU transport, dynamic batching, or any
performance improvement has already been implemented or measured.
