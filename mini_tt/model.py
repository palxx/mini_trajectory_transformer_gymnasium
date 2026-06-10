from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EinLinear(nn.Module):
    """Separate linear projection for each transition dimension.

    Input:  [B, transition_dim, in_features]
    Output: [B, transition_dim, out_features]
    """

    def __init__(self, n_models: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.n_models = int(n_models)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(n_models, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_models, out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("bni,nio->bno", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    observation_dim: int
    action_dim: int
    transition_dim: int
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    action_weight: float = 5.0
    reward_weight: float = 1.0
    value_weight: float = 1.0
    mask_value_tokens: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        # Same idea as Janner et al.'s code: do not let the model condition on
        # previous return/value estimate tokens, so it cannot exploit leaked labels.
        if config.mask_value_tokens:
            value_col = config.transition_dim - 1
            mask[:, value_col :: config.transition_dim] = False
            mask = torch.tril(mask)
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        head_dim = c // self.n_head
        k = self.key(x).view(b, t, self.n_head, head_dim).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(~self.mask[:, :, :t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Small Trajectory Transformer GPT.

    The vocabulary is shared across scalar bins, but tokens are offset by
    transition dimension before embedding, matching the original repository's
    design. The output head is dimension-specific via EinLinear.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.embedding_dim = config.n_embd

        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def configure_optimizers(self, learning_rate: float, weight_decay: float, betas=(0.9, 0.95)):
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, EinLinear)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")
        param_dict = {pn: p for pn, p in self.named_parameters()}
        missing = set(param_dict) - (decay | no_decay)
        if missing:
            raise RuntimeError(f"Parameters not assigned to decay/no_decay groups: {missing}")
        return torch.optim.AdamW(
            [
                {"params": [param_dict[p] for p in sorted(decay)], "weight_decay": weight_decay},
                {"params": [param_dict[p] for p in sorted(no_decay)], "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )

    def offset_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        n_transitions = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim, device=idx.device) * self.vocab_size
        offsets = offsets.repeat(n_transitions)[:t]
        offset_idx = idx + offsets
        offset_idx = offset_idx.clone()
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_transition(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        b, t, c = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        if n_pad:
            pad = torch.zeros(b, n_pad, c, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x = x.view(-1, self.transition_dim, c)
        return x, n_pad

    def _target_weights(self, t: int, device: torch.device) -> torch.Tensor:
        base = torch.cat(
            [
                torch.ones(self.observation_dim, device=device),
                torch.ones(self.action_dim, device=device) * self.config.action_weight,
                torch.ones(1, device=device) * self.config.reward_weight,
                torch.ones(1, device=device) * self.config.value_weight,
            ]
        )
        n = int(np.ceil((t + 1) / self.transition_dim))
        # Targets are one token to the right of inputs, hence [1:t+1].
        return base.repeat(n)[1 : t + 1]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.block_size}")

        token_embeddings = self.tok_emb(self.offset_tokens(idx))
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        x_pad, n_pad = self.pad_to_transition(x)
        logits = self.head(x_pad).reshape(b, t + n_pad, self.vocab_size + 1)[:, :t]

        loss = None
        if targets is not None:
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(b, t)
            if mask is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            weights = self._target_weights(t, idx.device).view(1, t)
            weighted = loss_per_token * weights * mask.float()
            denom = (weights * mask.float()).sum().clamp_min(1.0)
            loss = weighted.sum() / denom
        return logits, loss
