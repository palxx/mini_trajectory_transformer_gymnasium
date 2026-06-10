from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .discretizer import QuantileDiscretizer


def discounted_cumsum(rewards: np.ndarray, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + discount * running
        out[t] = running
    return out[:, None]


@dataclass
class DatasetStats:
    dataset_id: str
    episodes: int
    transitions: int
    observation_dim: int
    action_dim: int
    transition_dim: int
    sequence_length: int
    n_bins: int


class MinariTrajectoryDataset(Dataset):
    """Offline trajectory dataset for the small Trajectory Transformer.

    Loads a Minari dataset, segments by episode, builds transitions
    [obs, action, reward, return_to_go], and discretizes scalar dimensions.

    This implementation assumes Box observations and Box actions, which matches
    the MuJoCo datasets used in the original Trajectory Transformer experiments.
    """

    def __init__(
        self,
        dataset_id: str = "mujoco/halfcheetah/medium-v0",
        sequence_length: int = 10,
        n_bins: int = 32,
        discount: float = 0.99,
        max_episodes: Optional[int] = 50,
        max_steps_per_episode: Optional[int] = None,
        stride: int = 1,
        download: bool = True,
        termination_penalty: Optional[float] = None,
    ) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        import gymnasium as gym
        import minari

        self.dataset_id = dataset_id
        self.sequence_length = int(sequence_length)
        self.n_bins = int(n_bins)
        self.discount = float(discount)
        self.minari_dataset = minari.load_dataset(dataset_id, download=download)

        self.paths: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []
        obs_dim: Optional[int] = None
        act_dim: Optional[int] = None

        for ep_i, episode in enumerate(self.minari_dataset.iterate_episodes()):
            if max_episodes is not None and ep_i >= max_episodes:
                break
            print(f'episode.observations', episode.observations.shape)
            obs = np.asarray(episode.observations)
            actions = np.asarray(episode.actions)
            rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1, 1)
            terminations = np.asarray(episode.terminations).reshape(-1)

            if obs.dtype == object or actions.dtype == object:
                raise TypeError("Dictionary or object observations/actions are not supported by this compact example.")
            if actions.ndim == 1:
                actions = actions[:, None]
            if obs.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"Expected 2D obs/actions, got obs {obs.shape}, actions {actions.shape}")

            # Minari episodes normally store T+1 observations and T actions.
            horizon = min(actions.shape[0], rewards.shape[0])
            obs = obs[:horizon]
            actions = actions[:horizon]
            rewards = rewards[:horizon]
            terminations = terminations[:horizon]

            if termination_penalty is not None and len(terminations) > 0:
                terminal_mask = terminations.astype(bool)
                rewards[terminal_mask] = float(termination_penalty)

            if max_steps_per_episode is not None:
                horizon = min(horizon, int(max_steps_per_episode))
                obs = obs[:horizon]
                actions = actions[:horizon]
                rewards = rewards[:horizon]

            if len(obs) < sequence_length:
                continue

            returns_to_go = discounted_cumsum(rewards, discount)
            joined = np.concatenate([obs, actions, rewards, returns_to_go], axis=-1).astype(np.float32)

            obs_dim = obs.shape[-1] if obs_dim is None else obs_dim
            act_dim = actions.shape[-1] if act_dim is None else act_dim
            if obs.shape[-1] != obs_dim or actions.shape[-1] != act_dim:
                raise ValueError("All episodes must have the same observation and action dimensions.")

            path_id = len(self.paths)
            self.paths.append(joined)
            for start in range(0, len(joined) - sequence_length + 1, stride):
                self.indices.append((path_id, start))

        if not self.paths:
            raise ValueError(
                "No usable episodes were loaded. Check the dataset id, max_episodes, "
                "and sequence_length."
            )

        self.joined_raw = np.concatenate(self.paths, axis=0)
        self.discretizer = QuantileDiscretizer.fit(self.joined_raw, n_bins=n_bins)
        self.observation_dim = int(obs_dim)  # type: ignore[arg-type]
        self.action_dim = int(act_dim)  # type: ignore[arg-type]
        self.transition_dim = int(self.joined_raw.shape[-1])

        # Basic action-space check. Planning clips actions using the recovered env.
        env = self.minari_dataset.recover_environment()
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("This compact Trajectory Transformer expects a continuous Box action space.")
        env.close()

    @property
    def stats(self) -> DatasetStats:
        return DatasetStats(
            dataset_id=self.dataset_id,
            episodes=len(self.paths),
            transitions=int(sum(len(p) for p in self.paths)),
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            transition_dim=self.transition_dim,
            sequence_length=self.sequence_length,
            n_bins=self.n_bins,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_id, start = self.indices[idx]
        joined = self.paths[path_id][start : start + self.sequence_length]
        tokens = self.discretizer.discretize(joined).astype(np.int64).reshape(-1)
        x = torch.from_numpy(tokens[:-1]).long()
        y = torch.from_numpy(tokens[1:]).long()
        mask = torch.ones_like(x, dtype=torch.bool)
        return x, y, mask
