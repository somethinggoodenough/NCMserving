# Validation record

The executor was validated on 2026-10-23 against unmodified NCM generation at
upstream commit `c70274273f489abc73832b2b98584a8483b6b203`, using an RTX 4060,
PyTorch 2.6.0+cu118, FP32, and TF32 disabled.

- All ten group boundaries passed for CFG 0 and 1, four WMT14 samples per cell.
- At `split=5`, CFG 0 and 1 each passed all 3003 WMT14 test samples.
- Every sample matched all 313 concept codes, 64 final token IDs, and decoded text.
- CPU tests covered both toy-model boundaries, greedy and stochastic sampling,
  request interleaving, independent RNG, stale-state rejection, cancellation,
  unsupported configurations, and all-padding rejection.

Output hashes:

| CFG | Concept codes SHA-256 | Token IDs SHA-256 |
| --- | --- | --- |
| 0 | `836d1b0f0202ab73c03a1d00235fbec739cd4724fa807847f7bb9258f85a2ca4` | `c274657f93c9f1d603a7298817f05acf650d6308d9b6d8113b5fd6fc098612aa` |
| 1 | `33952bb4af0d0722e452d398610e3623bdd94f9d837f4fb3347a7b0518819e2a` | `25b671e0c53b6e530918cbc4f37141ba556af7e44d9b5555501caa0234eb17d5` |

The original source file used for that run has SHA-256
`66fc9cdebbc366f6e9b23615ad5ed439cc61b29297ba8c996fa3b7ebd0b024b4`.
This repository reorganizes that implementation as a package; the execution
flow and numerical operations are preserved.

After packaging, the standalone executor was run again against the pinned
upstream checkout on the same RTX 4060. Four fresh German prompts passed all
ten split points for CFG 0 and 1: 20/20 cells matched upstream concept codes
and final token IDs exactly. The correctness-only child process took 26.828
seconds including model loading; its local raw JSON is intentionally ignored.

These results establish correctness for the stated path. They do not establish
single-device speedup, multi-GPU scaling, transfer cost, or production readiness.
