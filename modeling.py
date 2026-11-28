from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PDMConfig:
    in_channels: int = 1
    patch: int = 1
    n_embd: int = 64
    n_layer: int = 4
    n_head: int = 4
    fno_modes: int = 64
    gla_chunk: int = 32
    ffn_hidden: int = 128
    dropout: float = 0.0
    use_gdn: bool = False
    gdn_every: int = 4
    n_classes: int = 0
    rul_head: bool = False
    seq_len: int = 2048

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0


class FNOSeqMixer(nn.Module):
    def __init__(self, cfg: PDMConfig):
        super().__init__()
        self.M = cfg.fno_modes
        self.filter_td = nn.Parameter(torch.empty(cfg.n_embd, cfg.fno_modes))
        self.out_scale = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        nn.init.normal_(self.filter_td, std=0.02)

    def forward(self, x):
        B, T, C = x.shape
        n_use = min(self.M, T)
        h = self.filter_td.new_zeros(2 * T, C)
        h[:n_use] = self.filter_td[:, :n_use].T
        xp = F.pad(x, (0, 0, 0, T))
        Y = torch.fft.rfft(xp, dim=1) * torch.fft.rfft(h, dim=0).unsqueeze(0)
        return self.out_scale(torch.fft.irfft(Y, n=2 * T, dim=1)[:, :T])


class SwiGLU(nn.Module):
    def __init__(self, cfg: PDMConfig):
        super().__init__()
        d, hd = (cfg.n_embd, cfg.ffn_hidden)
        self.gate = nn.Linear(d, hd, bias=False)
        self.up = nn.Linear(d, hd, bias=False)
        self.down = nn.Linear(hd, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class PDMBlock(nn.Module):
    def __init__(self, cfg: PDMConfig):
        super().__init__()
        self.mixer = FNOSeqMixer(cfg)
        self.ffn = SwiGLU(cfg)
        self.ln1 = nn.RMSNorm(cfg.n_embd)
        self.ln2 = nn.RMSNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FELAPDM(nn.Module):
    def __init__(self, cfg: PDMConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = cfg.patch
        self.embed = nn.Linear(cfg.in_channels * cfg.patch, cfg.n_embd)
        self.blocks = nn.ModuleList([PDMBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_out = nn.RMSNorm(cfg.n_embd)
        if cfg.n_classes > 0:
            self.cls_head = nn.Linear(cfg.n_embd, cfg.n_classes)
        if cfg.rul_head:
            self.rul_head = nn.Linear(cfg.n_embd, 1)

    def _patchify(self, x):
        B, T, Cin = x.shape
        p = self.patch
        if p > 1:
            T2 = T // p * p
            x = x[:, :T2].reshape(B, T2 // p, p * Cin)
        return x

    def _backbone(self, x):
        x = self._patchify(x)
        x = self.embed(x)
        x = F.rms_norm(x, (x.size(-1),))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_out(x)

    def forward(self, x, task: str = None):
        task = task or ("rul" if self.cfg.rul_head else "cls")
        h = self._backbone(x)
        if task == "cls":
            return self.cls_head(h.mean(dim=1))
        if task == "rul":
            return self.rul_head(h[:, -1, :]).squeeze(-1)
        raise ValueError(task)

    @torch.no_grad()
    def predict(self, x, task: str = None):
        self.eval()
        validate_window(x, self.cfg)
        out = self.forward(x, task=task)
        if (task or ("rul" if self.cfg.rul_head else "cls")) == "cls":
            probs = torch.softmax(out.float(), dim=-1)[0]
            idx = int(probs.argmax())
            return (idx, float(probs[idx]))
        return float(out.reshape(-1)[0])


def validate_window(x: torch.Tensor, cfg: PDMConfig):
    if x.dim() != 3:
        raise ValueError(
            f"expected a 3D tensor (batch, time, channels), got shape {tuple(x.shape)}"
        )
    if x.shape[-1] != cfg.in_channels:
        raise ValueError(
            f"expected {cfg.in_channels} channels in the last dimension, got {x.shape[-1]}. CWRU vibration is 1 channel; C-MAPSS is 14 sensors."
        )
    if x.shape[1] < cfg.patch:
        raise ValueError(
            f"time dimension {x.shape[1]} is shorter than the patch size {cfg.patch}"
        )


def preprocess_cwru(samples, expected_len: int = 2048) -> torch.Tensor:
    t = torch.as_tensor(samples, dtype=torch.float32).reshape(-1)
    if t.numel() != expected_len:
        raise ValueError(
            f"CWRU window must be {expected_len} samples (12 kHz), got {t.numel()}"
        )
    t = (t - t.mean()) / (t.std() + 1e-06)
    return t.reshape(1, expected_len, 1)


def preprocess_cmapss(cycles, sensor_min, sensor_max, window: int = 30) -> torch.Tensor:
    t = torch.as_tensor(cycles, dtype=torch.float32)
    if t.dim() != 2 or t.shape[1] != 14:
        raise ValueError(f"C-MAPSS input must be (window, 14), got {tuple(t.shape)}")
    if t.shape[0] != window:
        raise ValueError(f"C-MAPSS window must be {window} cycles, got {t.shape[0]}")
    lo = torch.as_tensor(sensor_min, dtype=torch.float32)
    hi = torch.as_tensor(sensor_max, dtype=torch.float32)
    t = (t - lo) / (hi - lo + 1e-06)
    return t.reshape(1, window, 14)


def _load_state(path: str):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


_CONFIG_FIELDS = set(PDMConfig.__dataclass_fields__.keys())


def _to_pdm_config(cfg_dict: dict, variant: str = None) -> PDMConfig:
    if "variants" in cfg_dict:
        variant = variant or cfg_dict.get("default_variant")
        if variant not in cfg_dict["variants"]:
            raise ValueError(
                f"unknown variant {variant!r}; choose one of {list(cfg_dict['variants'])}"
            )
        cfg_dict = cfg_dict["variants"][variant]
    return PDMConfig(**{k: v for k, v in cfg_dict.items() if k in _CONFIG_FIELDS})


def load_model(path_or_repo: str, config: dict = None, variant: str = None):
    cfg_dict = config
    weights_path = None
    if os.path.isdir(path_or_repo):
        cfg_dict = cfg_dict or _read_json(os.path.join(path_or_repo, "config.json"))
        v = variant or (
            cfg_dict.get("default_variant") if isinstance(cfg_dict, dict) else None
        )
        cand = os.path.join(path_or_repo, f"{v}.safetensors") if v else None
        if cand and os.path.isfile(cand):
            weights_path = cand
        else:
            weights_path = os.path.join(path_or_repo, "model.safetensors")
    elif os.path.isfile(path_or_repo):
        if path_or_repo.endswith(".safetensors"):
            beside = os.path.join(os.path.dirname(path_or_repo), "config.json")
            cfg_dict = cfg_dict or _read_json(beside)
            weights_path = path_or_repo
        else:
            ck = torch.load(path_or_repo, map_location="cpu", weights_only=False)
            cfg_dict = cfg_dict or ck["cfg"]
            model = FELAPDM(_to_pdm_config(cfg_dict, variant))
            model.load_state_dict(ck["model"])
            model.eval()
            return model
    else:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(path_or_repo, "config.json")
        cfg_dict = cfg_dict or _read_json(cfg_path)
        v = variant or (
            cfg_dict.get("default_variant") if isinstance(cfg_dict, dict) else None
        )
        try:
            weights_path = (
                hf_hub_download(path_or_repo, f"{v}.safetensors")
                if v
                else hf_hub_download(path_or_repo, "model.safetensors")
            )
        except Exception:
            weights_path = hf_hub_download(path_or_repo, "model.safetensors")
    model = FELAPDM(_to_pdm_config(cfg_dict, variant))
    model.load_state_dict(_load_state(weights_path))
    model.eval()
    return model


from_pretrained = load_model


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
