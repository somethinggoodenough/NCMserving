"""Exact single-device concept-level execution for the supported NCM path."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

from . import prefix_kv as kv


def release_on_error(method):
    """Release owned cache on execution failure, including direct API calls."""
    @wraps(method)
    def wrapped(self, state, *args, **kwargs):
        self._validate(state)
        try:
            return method(self, state, *args, **kwargs)
        except BaseException:
            self.abort(state)
            raise
    return wrapped


@dataclass
class LevelState:
    """All state required to resume an NCM request at a concept-level boundary."""

    request_id: str
    model_id: str
    cfg: float
    original_batch: int
    next_level: int
    prefix: torch.Tensor
    pad_mask: torch.Tensor
    pad_bias: torch.Tensor
    level_embedding: torch.Tensor
    latent: torch.Tensor
    codes: list[torch.Tensor]
    cpu_rng: torch.Tensor
    device_rng: torch.Tensor | None
    status: str = "active"
    prefix_kv: tuple | None = None

    def snapshot(self) -> "LevelState":
        """Create an ownership-independent logical handoff snapshot."""

        return copy.deepcopy(self)


class NCMLevelExecutor:
    """Execute one fixed batch through short and long concept-level groups.

    The executor supports the WMT14-style raw-prefix, predict-level0, eager
    RoPE path in the upstream NCM implementation. It deliberately does not
    claim multi-GPU transport, dynamic batching, or concurrent thread safety.
    """

    def __init__(self, runtime: Any, split: int = 5, *, prefix_cache=False, record_trace=True):
        self.runtime = runtime
        self.net = runtime.net
        self.vq = runtime.tokenizer.vqvae.vq
        self.levels = tuple(self.net.levels)
        if not 1 <= split < len(self.levels):
            raise ValueError("split must leave at least one concept level in each group")
        net = self.net
        if (
            not net.predict_level0
            or net.query_level0
            or net.use_ctx_encoder
            or not net.use_rope
            or net.use_flex_attention
            or not bool(getattr(net.config, "raw_token_prefix", False))
            or bool(getattr(net.config, "contextual_prefix", False))
        ):
            raise ValueError(
                "executor supports WMT14-style raw-prefix predict-level0 eager RoPE only"
            )
        if net.training or runtime.tokenizer.vqvae.training:
            raise ValueError("models must be in eval mode")
        self.device = next(net.parameters()).device
        self.prefix_cache = prefix_cache
        self.record_trace = record_trace
        if prefix_cache:
            kv.validate_blocks(net)
        self.split = split
        self.model_id = self._model_fingerprint()
        self.states: dict[str, LevelState] = {}
        self.trace: list[dict[str, Any]] = []

    def _model_fingerprint(self) -> str:
        digest = hashlib.sha256(
            repr((self.levels, str(self.net.config), str(self.runtime.cfg.generation))).encode()
        )
        for module in (self.net, self.runtime.tokenizer.vqvae):
            for name, tensor in module.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _devices(self) -> list[int]:
        return [self.device.index or 0] if self.device.type == "cuda" else []

    @torch.no_grad()
    def submit(
        self, request_id: str, prefix: torch.Tensor, *, seed: int, cfg: float = 0.0
    ) -> LevelState:
        if request_id in self.states:
            raise ValueError("duplicate request_id")
        if self.prefix_cache and torch.is_autocast_enabled(self.device.type):
            raise ValueError("prefix cache requires FP32 execution with autocast disabled")
        if cfg not in (0.0, 1.0):
            raise ValueError("CFG must be 0 or 1 for the validated execution path")
        if (
            prefix.ndim != 2
            or prefix.shape[0] == 0
            or prefix.shape[1] != self.net.prefix_len
            or prefix.dtype != torch.long
        ):
            raise ValueError("invalid prefix shape/dtype")
        if not torch.all((prefix != self.runtime.pad).any(dim=1)):
            raise ValueError("all-padding prompt")

        batch = prefix.shape[0]
        prefix = prefix.to(self.device)
        if cfg > 0:
            null = torch.full_like(prefix, self.runtime.pad)
            null[:, 0] = self.runtime.eos
            prefix = torch.cat([prefix, null], dim=0)

        embedding = self.net.transformer.wte(
            self.runtime.tokenizer.vqvae.encoder.wte(prefix)
        )
        pad_mask = prefix == self.runtime.pad
        pad_bias = torch.zeros(
            prefix.shape[0], 1, 1, prefix.shape[1], device=self.device
        )
        pad_bias.masked_fill_(pad_mask[:, None, None, :], -torch.inf)
        if self.net.config.use_level_encoding:
            level_embedding = self.net.transformer.wle(self.net.lvl_1L)
        else:
            level_embedding = torch.zeros(
                1,
                self.net.L - self.levels[0],
                self.net.config.n_embd,
                device=self.device,
            )

        with torch.random.fork_rng(devices=self._devices()):
            torch.random.default_generator.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.default_generators[self.device.index or 0].manual_seed(seed)
            cpu_rng = torch.get_rng_state().clone()
            device_rng = (
                torch.cuda.get_rng_state(self.device).clone()
                if self.device.type == "cuda"
                else None
            )

        state = LevelState(
            request_id=request_id,
            model_id=self.model_id,
            cfg=cfg,
            original_batch=batch,
            next_level=0,
            prefix=embedding,
            pad_mask=pad_mask,
            pad_bias=pad_bias,
            level_embedding=level_embedding,
            latent=torch.zeros(
                prefix.shape[0],
                self.levels[-1],
                self.runtime.tokenizer.vqvae.embed_dim,
                device=self.device,
            ),
            codes=[],
            cpu_rng=cpu_rng,
            device_rng=device_rng,
        )
        self.states[request_id] = state
        return state

    def _validate(self, state: LevelState) -> None:
        if state.model_id != self.model_id:
            raise ValueError("model/config mismatch")
        if self.states.get(state.request_id) is not state:
            raise ValueError("stale or foreign state")
        if state.status != "active":
            raise ValueError("request is not active")

    @torch.no_grad()
    def _step(self, state: LevelState, group: str) -> None:
        self._validate(state)
        net, level = self.net, state.next_level
        generation = self.runtime.cfg.generation
        with torch.random.fork_rng(devices=self._devices()):
            torch.set_rng_state(state.cpu_rng)
            if state.device_rng is not None:
                torch.cuda.set_rng_state(state.device_rng, self.device)
            if level == 0:
                hidden = state.prefix
                mask = (
                    torch.zeros(
                        1, 1, net.prefix_len, net.prefix_len, device=self.device
                    )
                    + state.pad_bias
                )
                rope = (net.rope_cos_gen_0, net.rope_sin_gen_0)
                if self.prefix_cache:
                    hidden, state.prefix_kv = kv.prefill(net, hidden, state.pad_bias)
                else:
                    for block in net.transformer.blocks:
                        hidden = block(hidden, attn_mask=mask, rope_cos_sin=rope)
                    hidden = net.transformer.ln_f(hidden)
                logits = net.lm_head_l0(
                    net._level0_slice(hidden, state.pad_mask)
                )
            else:
                embedded = self.vq.embeddings[level - 1](state.codes[-1])
                state.latent, inputs = self.vq.get_next_autoregressive_input(
                    level - 1, state.latent, embedded
                )
                if net.use_post_upsample_conv:
                    inputs = net._apply_post_upsample_conv_single(inputs, level - 1)
                offset = sum(self.levels[1:level])
                inputs = net.transformer.wte(inputs) + state.level_embedding[
                    :, offset : offset + self.levels[level], :
                ]
                if self.prefix_cache:
                    logits = kv.target_logits(net, inputs, state.prefix_kv, state.pad_bias, level - 1)
                else:
                    logits = net._decode_level_logits(
                        inputs, state.prefix, level - 1, state.pad_bias, hidx=level - 1
                    )
            if state.cfg > 0:
                weight = state.cfg * (level / max(len(self.levels) - 1, 1))
                logits = (
                    (1.0 + weight) * logits[: state.original_batch]
                    - weight * logits[state.original_batch :]
                )
            code = net._sample_level_raw(
                logits,
                float(generation.temperature),
                int(generation.top_k),
                float(generation.top_p),
            )
            if state.cfg > 0:
                code = torch.cat([code, code], dim=0)
            state.codes.append(code.detach().clone())
            state.cpu_rng = torch.get_rng_state().clone()
            if self.device.type == "cuda":
                state.device_rng = torch.cuda.get_rng_state(self.device).clone()

        state.next_level += 1
        if self.record_trace:
            self.trace.append(
                {
                    "request_id": state.request_id,
                    "group": group,
                    "level": level,
                    "concept_length": self.levels[level],
                    "batch_size": state.original_batch,
                }
            )

    @release_on_error
    def short_group(self, state: LevelState) -> LevelState:
        self._validate(state)
        if state.next_level != 0:
            raise ValueError("short group already executed")
        while state.next_level < self.split:
            self._step(state, "A")
        return state

    @release_on_error
    def handoff(self, state: LevelState) -> LevelState:
        self._validate(state)
        if state.next_level != self.split:
            raise ValueError("handoff must occur at configured split")
        snapshot = state.snapshot()
        state.status = "transferred"
        state.prefix_kv = None
        self.states[state.request_id] = snapshot
        return snapshot

    @torch.no_grad()
    @release_on_error
    def long_group(self, state: LevelState) -> dict[str, torch.Tensor]:
        self._validate(state)
        if state.next_level != self.split:
            raise ValueError("long group requires complete short group")
        while state.next_level < len(self.levels):
            self._step(state, "B")
        codes = torch.cat(state.codes, dim=1)[: state.original_batch]
        _, levels = self.vq.idx_to_zhats(codes, return_zlevels=True)
        latent = levels[-1].detach().clone()
        logits = self.runtime.tokenizer.vqvae.decoder(latent)
        state.status = "completed"
        state.prefix_kv = None
        result = {"codes": codes, "latent": latent, "token_ids": logits.argmax(-1)}
        del self.states[state.request_id]
        return result

    def cancel(self, state: LevelState) -> None:
        self._validate(state)
        state.status = "cancelled"
        state.prefix_kv = None
        del self.states[state.request_id]

    def abort(self, state: LevelState) -> None:
        """Release an owned failed request without attempting to resume it."""
        if self.states.get(state.request_id) is state:
            state.prefix_kv = None
            state.status = 'failed'
            del self.states[state.request_id]
