"""Public serving facade for synchronous single-device requests."""

from __future__ import annotations

from uuid import uuid4

import torch

from .engine import NCMLevelExecutor
from .scheduler import TwoGroupScheduler


class NCMServingSystem:
    def __init__(self, runtime, *, split: int = 5, prefix_cache=False, record_trace=True):
        self.runtime = runtime
        self.executor = NCMLevelExecutor(runtime, split=split, prefix_cache=prefix_cache,
                                         record_trace=record_trace)
        self.scheduler = TwoGroupScheduler(self.executor)

    def submit_prefixes(
        self,
        prefixes: torch.Tensor,
        *,
        seed: int,
        cfg: float = 0.0,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or uuid4().hex
        self.scheduler.submit(request_id, prefixes, seed=seed, cfg=cfg)
        return request_id

    def generate_prefixes(
        self,
        prefixes: torch.Tensor,
        *,
        seed: int,
        cfg: float = 0.0,
        request_id: str | None = None,
    ) -> dict[str, torch.Tensor]:
        request_id = self.submit_prefixes(
            prefixes, seed=seed, cfg=cfg, request_id=request_id
        )
        return self.scheduler.run_until_complete(request_id)

    def generate_text(
        self,
        prompts: list[str],
        *,
        seed: int,
        cfg: float = 0.0,
        request_id: str | None = None,
    ) -> dict:
        prefixes = self.runtime.encode(prompts)
        result = self.generate_prefixes(
            prefixes, seed=seed, cfg=cfg, request_id=request_id
        )
        return {
            "texts": self.runtime.decode(result["token_ids"]),
            "token_ids": result["token_ids"].detach().cpu(),
            "codes": result["codes"].detach().cpu(),
        }
