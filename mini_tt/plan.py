from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .discretizer import QuantileDiscretizer
from .model import GPT, GPTConfig
from .utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan/evaluate with a small Gymnasium Trajectory Transformer.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset id stored in the checkpoint.")
    parser.add_argument("--download", action="store_true", help="Download Minari dataset if needed.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)

    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--logprob_weight", type=float, default=0.01)
    parser.add_argument("--context_transitions", type=int, default=0)
    parser.add_argument(
        "--target_return",
        type=float,
        default=None,
        help=(
            "Decision-Transformer-style conditioning: desired total return for the episode. "
            "Instead of storing the model's own predicted return-to-go in the rolling context, "
            "store (target_return minus return collected so far), discounted appropriately, so "
            "future planning steps are conditioned on reaching that target. Has no effect unless "
            "--context_transitions > 0, since that is what carries the value token into context."
        ),
    )
    parser.add_argument("--record_dir", type=str, default=None,
                        help="Directory to save per-episode GIF animations. e.g. runs/videos")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: str | Path, device: str):
    # weights_only=False is needed because this checkpoint stores numpy arrays in
    # the discretizer state. Load checkpoints you trust.
    state = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**state["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    discretizer = QuantileDiscretizer.from_state_dict(state["discretizer"])
    return state, model, discretizer


@torch.no_grad()
def beam_plan(
    model: GPT,
    discretizer: QuantileDiscretizer,
    observation: np.ndarray,
    context_tokens: list[int],
    horizon: int,
    beam_width: int,
    topk: int,
    temperature: float,
    discount: float,
    logprob_weight: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, list[int]]:
    obs_dim = model.observation_dim
    act_dim = model.action_dim
    trans_dim = model.transition_dim

    obs_tokens = discretizer.discretize(np.asarray(observation, dtype=np.float32), dims=range(obs_dim)).reshape(-1)
    prefix = list(context_tokens) + obs_tokens.tolist()
    context_len = len(context_tokens)
    target_total_len = context_len + horizon * trans_dim
    n_new = target_total_len - len(prefix)
    if n_new <= 0:
        raise ValueError("Prefix is already longer than the planned sequence. Reduce context_transitions or increase horizon.")

    start = torch.tensor(prefix, dtype=torch.long, device=device).view(1, -1)
    beams: list[tuple[torch.Tensor, float]] = [(start, 0.0)]

    for _ in range(n_new):
        candidates: list[tuple[torch.Tensor, float]] = []
        for seq, logp in beams:
            idx = seq[:, -model.block_size :]
            logits, _ = model(idx)
            # Only generate real bin tokens [0, vocab_size-1], not the stop token.
            next_logits = logits[0, -1, : model.vocab_size] / max(temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            k = min(topk, probs.numel())
            values, tokens = torch.topk(probs, k=k)
            for p, tok in zip(values, tokens):
                new_seq = torch.cat([seq, tok.view(1, 1)], dim=1)
                candidates.append((new_seq, logp + float(torch.log(p.clamp_min(1e-12)).item())))
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_width]

    best_score = -float("inf")
    best_traj = None
    best_tokens: Optional[np.ndarray] = None
    best_logp = -float("inf")

    for seq, logp in beams:
        full_tokens = seq[0].detach().cpu().numpy().astype(np.int64)
        plan_tokens = full_tokens[context_len : context_len + horizon * trans_dim]
        if plan_tokens.shape[0] != horizon * trans_dim:
            continue
        plan_tokens_2d = plan_tokens.reshape(horizon, trans_dim)
        traj = discretizer.reconstruct(plan_tokens_2d)
        rewards = traj[:, obs_dim + act_dim]
        last_value = traj[-1, -1]
        reward_score = float(np.sum((discount ** np.arange(horizon)) * rewards) + (discount ** horizon) * last_value)
        score = reward_score + logprob_weight * (logp / max(n_new, 1))
        if score > best_score:
            best_score = score
            best_traj = traj
            best_tokens = plan_tokens_2d
            best_logp = logp

    if best_traj is None or best_tokens is None:
        raise RuntimeError("Beam search failed to produce a complete trajectory.")

    action = best_traj[0, obs_dim : obs_dim + act_dim]
    return action.astype(np.float32), best_traj.astype(np.float32), float(best_score), best_tokens.reshape(-1).tolist()


def normalized_score(minari_dataset, total_return: float) -> float:
    try:
        import minari

        score = minari.get_normalized_score(minari_dataset, np.asarray([total_return], dtype=np.float64))
        return float(np.asarray(score).reshape(-1)[0])
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    state, model, discretizer = load_checkpoint(args.checkpoint, device)

    import gymnasium as gym  # noqa: F401 - makes the Gymnasium dependency explicit.
    import minari

    dataset_id = args.dataset or state["dataset_id"]
    discount = float(args.discount if args.discount is not None else state.get("discount", 0.99))
    minari_dataset = minari.load_dataset(dataset_id, download=args.download)
    render_kwargs = {"render_mode": "rgb_array"} if args.record_dir else {}
    try:
        env = minari_dataset.recover_environment(eval_env=True, **render_kwargs)
    except TypeError:
        env = minari_dataset.recover_environment(**render_kwargs)

    if args.max_steps is not None:
        max_steps = args.max_steps
    elif getattr(env, "spec", None) is not None and env.spec is not None and env.spec.max_episode_steps is not None:
        max_steps = int(env.spec.max_episode_steps)
    else:
        max_steps = 1000

    if args.target_return is not None and args.context_transitions <= 0:
        print(
            "Warning: --target_return only conditions planning through the carried-over "
            "context, so it has no effect unless --context_transitions > 0."
        )

    returns = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        obs = np.asarray(obs, dtype=np.float32)
        total_return = 0.0
        context_tokens: list[int] = []
        remaining_target = args.target_return
        frames: list[np.ndarray] = []
        if args.record_dir:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        for t in range(max_steps):
            action, planned_traj, planned_score, planned_tokens = beam_plan(
                model=model,
                discretizer=discretizer,
                observation=obs,
                context_tokens=context_tokens,
                horizon=args.horizon,
                beam_width=args.beam_width,
                topk=args.topk,
                temperature=args.temperature,
                discount=discount,
                logprob_weight=args.logprob_weight,
                device=device,
            )
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(env.action_space.dtype)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            if args.record_dir:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if args.context_transitions > 0:
                # Store the actual observed transition. For the value token, prefer a
                # user-specified target return-to-go (Decision-Transformer style) over
                # the model's own predicted RTG, so future planning steps are conditioned
                # on reaching that target rather than on the model's self-estimate.
                if remaining_target is not None:
                    value_est = remaining_target
                else:
                    value_est = float(planned_traj[0, -1])
                actual_transition = np.concatenate(
                    [
                        obs.reshape(-1),
                        np.asarray(action, dtype=np.float32).reshape(-1),
                        np.asarray([reward, value_est], dtype=np.float32),
                    ]
                )
                new_tokens = discretizer.discretize(actual_transition).reshape(-1).tolist()
                context_tokens.extend(new_tokens)
                max_context_tokens = args.context_transitions * model.transition_dim
                context_tokens = context_tokens[-max_context_tokens:]

            if args.verbose:
                target_msg = f" target_rtg={remaining_target:.3f}" if remaining_target is not None else ""
                print(
                    f"ep={ep} t={t} reward={float(reward):.3f} return={total_return:.3f} "
                    f"terminated={terminated} truncated={truncated} plan_score={planned_score:.3f}{target_msg}"
                )

            if remaining_target is not None:
                # rtg_t = r_t + discount * rtg_{t+1}  =>  rtg_{t+1} = (rtg_t - r_t) / discount
                remaining_target = (remaining_target - float(reward)) / max(discount, 1e-6)

            obs = np.asarray(next_obs, dtype=np.float32)
            if terminated or truncated:
                break

        score = normalized_score(minari_dataset, total_return)
        returns.append(total_return)
        print(f"episode {ep}: return={total_return:.3f} normalized_score={score:.3f} steps={t + 1}")

        if args.record_dir and frames:
            import imageio
            rec_dir = Path(args.record_dir)
            rec_dir.mkdir(parents=True, exist_ok=True)
            gif_path = rec_dir / f"episode_{ep}.gif"
            imageio.mimsave(str(gif_path), frames, fps=30)
            print(f"Saved animation: {gif_path}")

    env.close()
    print(f"mean_return={np.mean(returns):.3f} std_return={np.std(returns):.3f}")


if __name__ == "__main__":
    main()
