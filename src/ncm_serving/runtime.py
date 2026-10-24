"""Adapter for loading the pinned upstream NCM implementation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf


SUPPORTED_NCM_SHA = "c70274273f489abc73832b2b98584a8483b6b203"


@dataclass
class NCMRuntime:
    root: Path
    cfg: object
    tokenizer: object
    net: object
    eos: int
    pad: int

    def encode(self, prompts: Iterable[str]) -> torch.Tensor:
        length = int(self.cfg.model.prefix_len)
        rows = []
        for prompt in prompts:
            ids = list(self.tokenizer.pre_tokenizer.hf_tokenizer(prompt)["input_ids"])
            rows.append((ids[:length] + [self.pad] * length)[:length])
        if not rows:
            raise ValueError("at least one prompt is required")
        return torch.tensor(rows, dtype=torch.long, device="cpu")

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        texts = []
        for row in token_ids.detach().cpu().tolist():
            if self.eos in row:
                row = row[: row.index(self.eos)]
            row = [token for token in row if token not in (self.eos, self.pad)]
            texts.append(self.tokenizer.pre_tokenizer.decode(row, skip_special_tokens=True))
        return texts


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def verify_ncm_checkout(root: Path, expected_sha: str = SUPPORTED_NCM_SHA) -> str:
    required = [root / "models/next_concept.py", root / "tokenization/vqvae_tokenizer.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"not an NCM checkout; missing: {', '.join(missing)}")
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if expected_sha and actual != expected_sha:
        raise ValueError(f"unsupported NCM revision {actual}; expected {expected_sha}")
    return actual


def load_wmt14_runtime(
    ncm_root: str | Path,
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    device: str = "cuda",
    expected_sha: str = SUPPORTED_NCM_SHA,
) -> NCMRuntime:
    """Load the validated WMT14 runtime from an external NCM checkout."""

    root = Path(ncm_root).resolve()
    checkpoints = Path(checkpoint_dir).resolve()
    local_tokenizer = Path(tokenizer_dir).resolve()
    verify_ncm_checkout(root, expected_sha)
    configure_torch()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(root / "config/eval/wmt14.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = str(checkpoints / "tokenizer.pt")
    cfg.tokenizer.pre_tokenizer.model_id = str(local_tokenizer)
    tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
    tokenizer.vqvae = tokenizer.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tokenizer.vqvae, config=cfg.model).to(device)
    state = torch.load(checkpoints / "generator.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"], strict=True)
    net.eval()
    return NCMRuntime(
        root=root,
        cfg=cfg,
        tokenizer=tokenizer,
        net=net,
        eos=int(tokenizer.pre_tokenizer.eos_token_id),
        pad=int(tokenizer.pre_tokenizer.pad_token_id),
    )
