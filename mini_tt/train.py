from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import MinariTrajectoryDataset
from .model import GPT, GPTConfig
from .utils import ensure_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Gymnasium/Minari Trajectory Transformer.")
    parser.add_argument("--dataset", type=str, default="mujoco/halfcheetah/medium-v0")
    parser.add_argument("--download", action="store_true", help="Download the Minari dataset if it is not local.")
    parser.add_argument("--out_dir", type=str, default="runs/mini_tt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # Smaller-scale knobs.
    parser.add_argument("--max_episodes", type=int, default=50, help="Use fewer episodes for a small local run.")
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--termination_penalty", type=float, default=None)

    # GPT architecture. Original repo used a deeper/wider setting; these defaults are small.
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--action_weight", type=float, default=5.0)
    parser.add_argument("--reward_weight", type=float, default=1.0)
    parser.add_argument("--value_weight", type=float, default=1.0)

    # Optimization.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument(
        "--render_episodes",
        type=str,
        default=None,
        help="Comma-separated dataset episode indices to render as GIFs, e.g. '3,4,5'.",
    )
    parser.add_argument(
        "--render_dir",
        type=str,
        default=None,
        help="Output directory for episode GIFs (default: <out_dir>/episode_renders).",
    )
    parser.add_argument("--render_fps", type=int, default=30)
    return parser.parse_args()


def save_checkpoint(path: Path, model: GPT, dataset: MinariTrajectoryDataset, args: argparse.Namespace, step: int) -> None:
    checkpoint = {
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "discretizer": dataset.discretizer.state_dict(),
        "dataset_id": dataset.dataset_id,
        "discount": dataset.discount,
        "step": step,
        "args": vars(args),
        "stats": dataset.stats.__dict__,
    }
    torch.save(checkpoint, path)


def render_dataset_episodes(
    dataset: MinariTrajectoryDataset,
    episode_indices: list[int],
    out_dir: Path,
    fps: int,
) -> None:
    import imageio

    minari_dataset = dataset.minari_dataset
    env = minari_dataset.recover_environment(render_mode="rgb_array")
    env.reset()
    unwrapped = env.unwrapped
    nq = unwrapped.model.nq
    nv = unwrapped.model.nv
    # MuJoCo locomotion envs commonly exclude the leading qpos entries (e.g. x position)
    # from observations; pad them back with zeros so set_state has the right shapes.
    n_excluded = max(nq + nv - dataset.observation_dim, 0)

    out_dir = ensure_dir(out_dir)
    for episode in minari_dataset.iterate_episodes(episode_indices=episode_indices):
        frames = []
        for obs in np.asarray(episode.observations, dtype=np.float64):
            qpos = np.concatenate([np.zeros(n_excluded), obs[: nq - n_excluded]])
            qvel = obs[nq - n_excluded :]
            unwrapped.set_state(qpos, qvel)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if frames:
            gif_path = out_dir / f"episode_{episode.id}.gif"
            imageio.mimsave(str(gif_path), frames, fps=fps)
            print(f"Saved dataset episode animation: {gif_path}")
    env.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    print(f"Loading Minari dataset: {args.dataset}")
    dataset = MinariTrajectoryDataset(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        n_bins=args.n_bins,
        discount=args.discount,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        stride=args.stride,
        download=args.download,
        termination_penalty=args.termination_penalty,
    )
    stats = dataset.stats
    print(
        f"Dataset windows: {len(dataset)} | episodes: {stats.episodes} | transitions: {stats.transitions} | "
        f"obs_dim: {stats.observation_dim} | act_dim: {stats.action_dim} | transition_dim: {stats.transition_dim}"
    )

    if args.render_episodes:
        episode_indices = [int(i) for i in args.render_episodes.split(",") if i.strip() != ""]
        render_dir = Path(args.render_dir) if args.render_dir else out_dir / "episode_renders"
        render_dataset_episodes(dataset, episode_indices, render_dir, fps=args.render_fps)

    block_size = args.sequence_length * dataset.transition_dim - 1
    config = GPTConfig(
        vocab_size=args.n_bins,
        block_size=block_size,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        transition_dim=dataset.transition_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        embd_pdrop=args.embd_pdrop,
        resid_pdrop=args.resid_pdrop,
        attn_pdrop=args.attn_pdrop,
        action_weight=args.action_weight,
        reward_weight=args.reward_weight,
        value_weight=args.value_weight,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(args.learning_rate, args.weight_decay)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=True,
    )

    save_json(out_dir / "config.json", {"args": vars(args), "stats": stats.__dict__, "model_config": config.to_dict()})

    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y, mask in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            _, loss = model(x, y, mask)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()

            global_step += 1
            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / max(n_batches, 1):.4f}")

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        if epoch % args.save_every == 0:
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(ckpt_path, model, dataset, args, global_step)
            save_checkpoint(out_dir / "checkpoint_latest.pt", model, dataset, args, global_step)
            print(f"Saved {ckpt_path}")

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    save_checkpoint(out_dir / "checkpoint_latest.pt", model, dataset, args, global_step)
    print(f"Done. Steps: {global_step} | elapsed: {(time.time() - t0):.1f}s | latest: {out_dir / 'checkpoint_latest.pt'}")


if __name__ == "__main__":
    main()
