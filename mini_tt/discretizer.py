from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class QuantileDiscretizer:
    """Per-dimension quantile tokenizer for continuous trajectory vectors.

    This mirrors the original Trajectory Transformer idea: every scalar in
    (observation, action, reward, return-to-go) is converted into a categorical
    token. Each transition dimension has its own quantile bins.
    """

    edges: np.ndarray
    centers: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    n_bins: int

    @classmethod
    def fit(cls, data: np.ndarray, n_bins: int = 32) -> "QuantileDiscretizer":
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data [N, D], got shape {data.shape}")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")

        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)[1:-1]

        edges = np.stack([np.quantile(data[:, d], qs) for d in range(data.shape[1])], axis=0)
        edges = np.maximum.accumulate(edges, axis=1)

        centers = []
        for d in range(data.shape[1]):
            bounds = np.concatenate([[mins[d]], edges[d], [maxs[d]]]).astype(np.float64)
            c = 0.5 * (bounds[:-1] + bounds[1:])
            # Constant dimensions produce identical boundaries. Keep a stable center.
            c = np.nan_to_num(c, nan=mins[d], posinf=maxs[d], neginf=mins[d])
            centers.append(c)
        centers = np.stack(centers, axis=0)
        return cls(edges=edges, centers=centers, mins=mins, maxs=maxs, n_bins=n_bins)

    @property
    def dim(self) -> int:
        return int(self.centers.shape[0])

    def _dims(self, dims: Iterable[int] | None, local_dim: int) -> list[int]:
        if dims is None:
            dims_list = list(range(local_dim))
        else:
            dims_list = list(dims)
        if len(dims_list) != local_dim:
            raise ValueError(f"dims length {len(dims_list)} does not match x last dim {local_dim}")
        if min(dims_list, default=0) < 0 or max(dims_list, default=0) >= self.dim:
            raise ValueError(f"dims must be in [0, {self.dim})")
        return dims_list

    def discretize(self, x: np.ndarray, dims: Iterable[int] | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        original_shape = x.shape
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim < 2:
            raise ValueError(f"Expected at least 1D/2D input, got shape {original_shape}")

        flat = x.reshape(-1, x.shape[-1])
        dims_list = self._dims(dims, flat.shape[-1])
        out = np.empty(flat.shape, dtype=np.int64)
        for local_i, global_i in enumerate(dims_list):
            out[:, local_i] = np.searchsorted(self.edges[global_i], flat[:, local_i], side="left")
        out = np.clip(out, 0, self.n_bins - 1)
        out = out.reshape(x.shape)
        return out[0] if len(original_shape) == 1 else out

    def reconstruct(self, tokens: np.ndarray, dims: Iterable[int] | None = None) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        original_shape = tokens.shape
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        flat = tokens.reshape(-1, tokens.shape[-1])
        dims_list = self._dims(dims, flat.shape[-1])
        out = np.empty(flat.shape, dtype=np.float32)
        for local_i, global_i in enumerate(dims_list):
            clipped = np.clip(flat[:, local_i], 0, self.n_bins - 1)
            out[:, local_i] = self.centers[global_i, clipped]
        out = out.reshape(tokens.shape)
        return out[0] if len(original_shape) == 1 else out

    def state_dict(self) -> dict:
        return {
            "edges": self.edges,
            "centers": self.centers,
            "mins": self.mins,
            "maxs": self.maxs,
            "n_bins": self.n_bins,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "QuantileDiscretizer":
        return cls(
            edges=np.asarray(state["edges"]),
            centers=np.asarray(state["centers"]),
            mins=np.asarray(state["mins"]),
            maxs=np.asarray(state["maxs"]),
            n_bins=int(state["n_bins"]),
        )
