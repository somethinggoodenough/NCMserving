# Design

NCM generation has an ordered hierarchy of concept levels. Every level still
runs the shared Transformer; the grouping boundary changes execution ownership,
not model semantics.

```text
submit -> short queue -> group A -> state handoff -> long queue -> group B -> decode
```

The handoff state owns the embedded prompt, padding mask and bias, accumulated
latent, all sampled codes, the next level index, model/config fingerprint, and
per-request CPU/CUDA RNG state. The old state becomes invalid at handoff.

The current scheduler runs on one device and keeps a fixed batch per request.
It prioritizes the long queue to release transferred state. The split is
configurable; `split=5` maps WMT14 levels `[1,4,8,14,20]` to A and
`[26,32,40,48,56,64]` to B. That default is validated for correctness but has
not been shown to be performance-optimal.

The NCM adapter requires an external checkout at upstream commit
`c70274273f489abc73832b2b98584a8483b6b203`. This repository does not vendor
upstream NCM, model weights, tokenizer data, or evaluation datasets.

Real multi-GPU workers, transport serialization, dynamic batching, admission
control, CUDA Graph capture, and online SLO scheduling remain future work.
