"""Direct signal decisions using a frozen decoder and explicit readout semantics.

This module contains no dataset names, split logic, experiment numbers or CLI.
Frozen parameters do not imply a no_grad backbone: gradients must reach the
trainable signal adapter through the fixed transformer during training.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def event_nll(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Per-observation negative log likelihood of a nonempty allowed event."""
    if logits.ndim != 2 or allowed.shape != logits.shape or allowed.dtype != torch.bool:
        raise ValueError("Need [batch, classes] logits and a matching boolean event mask")
    if not torch.isfinite(logits).all() or not allowed.any(dim=1).all():
        raise ValueError("Nonfinite logits or an empty annotation event")
    z = logits.float()
    return torch.logsumexp(z, dim=1) - torch.logsumexp(z.masked_fill(~allowed, -torch.inf), dim=1)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden state / mask shape mismatch")
    count = mask.sum(1, keepdim=True)
    if not (count > 0).all():
        raise ValueError("An observation cannot have an empty attention mask")
    return (hidden.float() * mask.unsqueeze(-1)).sum(1) / count


class PatchAdapter(nn.Module):
    """Project contiguous physical signal patches; no truncation or padding."""
    def __init__(self, patch_size: int, hidden_size: int, channels: int = 1):
        super().__init__()
        if min(patch_size, hidden_size, channels) <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.patch_size, self.channels = patch_size, channels
        self.projection = nn.Linear(patch_size * channels, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.channels:
            raise ValueError("Signals must be [batch, samples, declared_channels]")
        if not torch.isfinite(x).all() or x.shape[1] == 0 or x.shape[1] % self.patch_size:
            raise ValueError("Nonfinite signal or sample count not divisible by patch_size")
        return self.projection(x.reshape(x.shape[0], -1, self.patch_size * self.channels))


class SignalDecisionModel(nn.Module):
    """One encoder, with semantic, indexed or exact-token probability readout.

    decoder accepts inputs_embeds/attention_mask/use_cache/return_dict and returns
    last_hidden_state (the HF decoder contract). token_embeddings and lm_head are
    frozen modules from the same checkpoint. A token readout requires an explicit
    prefix and suffix, both fixed before target use. No input labels are accepted.
    """
    def __init__(self, decoder: nn.Module, *, hidden_size: int, classes: int,
                 patch_size: int, readout: str, candidate_vectors: torch.Tensor | None = None,
                 token_embeddings: nn.Module | None = None, lm_head: nn.Module | None = None,
                 token_ids: Sequence[int] = (), prefix_ids: Sequence[int] = (),
                 suffix_ids: Sequence[int] = (), score_temperature: float = 1.0):
        super().__init__()
        if readout not in {"semantic", "indexed", "token"} or classes < 2:
            raise ValueError("Unknown readout or invalid class count")
        if not 0 < score_temperature < float("inf"):
            raise ValueError("Score temperature must be positive and finite")
        self.decoder = decoder.requires_grad_(False).eval()
        self.adapter = PatchAdapter(patch_size, hidden_size)
        self.readout, self.classes, self.tau = readout, classes, float(score_temperature)
        self.token_embeddings = token_embeddings
        self.lm_head = lm_head
        for module in (token_embeddings, lm_head):
            if module is not None:
                module.requires_grad_(False).eval()
        self.indexed = nn.Linear(hidden_size, classes) if readout == "indexed" else None
        if readout == "semantic":
            if candidate_vectors is None or candidate_vectors.shape != (classes, hidden_size):
                raise ValueError("Semantic readout requires one aligned vector per class")
            if not torch.isfinite(candidate_vectors).all():
                raise ValueError("Nonfinite candidate vectors")
            if len(torch.unique(candidate_vectors, dim=0)) != classes:
                raise ValueError("Duplicate candidate vectors would duplicate event mass")
        vectors = torch.empty(0, hidden_size) if candidate_vectors is None else candidate_vectors
        self.register_buffer("candidates", vectors.detach().float().clone())
        for name, values in (("token_ids", token_ids), ("prefix_ids", prefix_ids), ("suffix_ids", suffix_ids)):
            self.register_buffer(name, torch.tensor(list(values), dtype=torch.long))
        if readout == "token":
            if token_embeddings is None or lm_head is None or len(set(token_ids)) != classes:
                raise ValueError("Token readout requires the real embedding/head and distinct class tokens")
            if not prefix_ids or not suffix_ids:
                raise ValueError("Token context must be explicitly bound")

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        for module in (self.token_embeddings, self.lm_head):
            if module is not None:
                module.eval()
        return self

    def _inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.adapter(x.float())
        dtype = next(self.decoder.parameters()).dtype
        if self.readout == "token":
            prefix = self.token_embeddings(self.prefix_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
            suffix = self.token_embeddings(self.suffix_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
            patches = torch.cat((prefix, patches.to(prefix.dtype), suffix), dim=1)
        patches = patches.to(dtype)
        mask = torch.ones(patches.shape[:2], device=patches.device, dtype=torch.long)
        return patches, mask

    def forward(self, x: torch.Tensor, *, candidate_vectors: torch.Tensor | None = None) -> torch.Tensor:
        if candidate_vectors is not None and self.readout != "semantic":
            raise ValueError("Declared candidate variants require a semantic readout")
        inputs, mask = self._inputs(x)
        # Deliberately NOT under no_grad: the fixed decoder propagates adapter gradients.
        hidden = self.decoder(inputs_embeds=inputs, attention_mask=mask,
                              use_cache=False, return_dict=True).last_hidden_state
        if self.readout == "token":
            # Read the complete vocabulary projection before selecting every class logit.
            return self.lm_head(hidden[:, -1]).float().index_select(1, self.token_ids)
        pooled = masked_mean(hidden, mask)
        if self.readout == "indexed":
            return self.indexed(pooled)
        vectors = self.candidates if candidate_vectors is None else candidate_vectors
        if vectors.shape != self.candidates.shape or not torch.isfinite(vectors).all():
            raise ValueError("Candidate variant does not cover the same diagnostic events")
        if len(torch.unique(vectors, dim=0)) != self.classes:
            raise ValueError("Candidate variant duplicates an event vector")
        return pooled @ vectors.to(device=pooled.device, dtype=pooled.dtype).T / self.tau

    def adaptation_state(self) -> dict:
        """Store adapted state only; never duplicate frozen pretrained weights."""
        out = {"adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
               "candidates": self.candidates.detach().cpu(), "readout": self.readout,
               "tau": self.tau, "token_ids": self.token_ids.detach().cpu(),
               "prefix_ids": self.prefix_ids.detach().cpu(), "suffix_ids": self.suffix_ids.detach().cpu()}
        if self.indexed is not None:
            out["indexed"] = {k: v.detach().cpu() for k, v in self.indexed.state_dict().items()}
        return out

    def restore_adaptation(self, state: dict) -> None:
        if state["readout"] != self.readout or state["tau"] != self.tau:
            raise ValueError("Checkpoint readout/temperature is incompatible")
        for name in ("candidates", "token_ids", "prefix_ids", "suffix_ids"):
            current = getattr(self, name)
            other = state[name].to(current.device)
            if current.shape != other.shape or not torch.equal(current, other):
                raise ValueError(f"Checkpoint changed fixed prediction contract: {name}")
        self.adapter.load_state_dict(state["adapter"], strict=True)
        if self.indexed is not None:
            self.indexed.load_state_dict(state["indexed"], strict=True)


def encode_candidate_vectors(decoder: nn.Module, tokenizer: object,
                             descriptions: Sequence[str], device: torch.device) -> tuple[torch.Tensor, float]:
    """Compile an explicit, fixed description set without adapting the decoder."""
    if len(descriptions) < 2 or any(not s.strip() for s in descriptions) or len(set(descriptions)) != len(descriptions):
        raise ValueError("Need distinct nonempty source-declared fault descriptions")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    text = tokenizer(list(descriptions), padding=True, return_tensors="pt").to(device)
    decoder.eval()
    with torch.no_grad():
        encoded = decoder(**text, use_cache=False, return_dict=True).last_hidden_state
        vectors = masked_mean(encoded, text["attention_mask"]).detach()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return vectors, time.perf_counter() - started


def compile_semantic_variant(model: SignalDecisionModel, descriptions: Sequence[str],
                             snapshot: str | Path, device: torch.device) -> tuple[torch.Tensor, float]:
    """Compile a prespecified sensitivity candidate set on the source side."""
    from transformers import AutoTokenizer
    if model.readout != "semantic" or len(descriptions) != model.classes:
        raise ValueError("A semantic variant must retain the approved event vocabulary")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    return encode_candidate_vectors(model.decoder, tokenizer, descriptions, device)


def load_signal_model(spec: dict, device: torch.device) -> tuple[SignalDecisionModel, object]:
    """Load the exact LOCAL snapshot; no network, CPU or checkpoint fallback."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snapshot = Path(spec["snapshot"]).expanduser().resolve()
    if not snapshot.is_dir() or snapshot.name != spec["revision"]:
        raise FileNotFoundError("The declared complete revision snapshot is unavailable")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[spec["dtype"]]
    causal = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=False, torch_dtype=dtype, attn_implementation="eager").to(device)
    causal.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer lacks an explicit pad token; do not infer one")
    descriptions = spec["descriptions"]
    if len(descriptions) < 2 or len(set(descriptions)) != len(descriptions):
        raise ValueError("Need distinct source-declared fault descriptions")
    decoder, vectors = causal.base_model, None
    compilation_seconds = None
    prefix, suffix, codes = [], [], []
    if spec["readout"] == "semantic":
        vectors, compilation_seconds = encode_candidate_vectors(decoder, tokenizer, descriptions, device)
        assignment = spec["assignment"]
        if sorted(assignment) != list(range(len(descriptions))):
            raise ValueError("Candidate assignment must be a bijection")
        vectors = vectors[assignment]
    elif spec["readout"] == "token":
        definitions = "\n".join(f"{code}: {description}" for code, description in zip(spec["codes"], descriptions))
        prefix_text = "Classify the bearing vibration using these definitions:\n" + definitions + "\nVibration:\n"
        suffix_text = "\nAnswer:"
        prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
        for code in spec["codes"]:
            full = tokenizer.encode(suffix_text + " " + code, add_special_tokens=False)
            if full[:-1] != suffix:
                raise ValueError(f"Code {code!r} is not one token at the actual response boundary")
            codes.append(full[-1])
    model = SignalDecisionModel(decoder, hidden_size=causal.config.hidden_size,
        classes=len(descriptions), patch_size=spec["patch_size"], readout=spec["readout"],
        candidate_vectors=vectors, token_embeddings=causal.get_input_embeddings(),
        lm_head=causal.get_output_embeddings(), token_ids=codes, prefix_ids=prefix,
        suffix_ids=suffix, score_temperature=spec["temperature"]).to(device)
    object.__setattr__(model, "_causal_lm", causal)
    model.candidate_encoding_seconds = compilation_seconds
    return model, causal
