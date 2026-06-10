# Mini Trajectory Transformer with Gymnasium + Minari

This is a compact, modernized Trajectory Transformer implementation inspired by
Janner et al.'s `trajectory-transformer` repository.

It keeps the same main components:

- offline trajectory dataset
- quantile discretization of continuous trajectory scalars
- GPT/minGPT-style causal transformer
- dimension-specific token offsets and output heads
- action/reward/value loss weighting
- beam-search planning

The old stack used `gym==0.18.0`, `mujoco-py`, and D4RL. This version uses:

- `gymnasium`
- `minari` for offline RL datasets
- modern PyTorch
- Gymnasium's `reset -> (obs, info)` and `step -> (obs, reward, terminated, truncated, info)` API

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Download a Minari dataset

Training can download automatically with `--download`, but you can also do it explicitly:

```bash
minari download mujoco/halfcheetah/medium-v0
```

`mujoco/halfcheetah/medium-v0` is the Minari/Gymnasium-style replacement for the
original code's common `halfcheetah-medium-v2` D4RL workflow. For a very small
local experiment, keep the same dataset but limit the number of episodes.

## Train a small model

```bash
python -m mini_tt.train \
  --dataset mujoco/halfcheetah/medium-v0 \
  --download \
  --max_episodes 50 \
  --sequence_length 10 \
  --n_bins 32 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --batch_size 128 \
  --epochs 5 \
  --device auto \
  --out_dir runs/halfcheetah_small
```

For an even smaller sanity check:

```bash
python -m mini_tt.train \
  --dataset mujoco/halfcheetah/medium-v0 \
  --download \
  --max_episodes 5 \
  --max_train_steps 100 \
  --epochs 1 \
  --out_dir runs/debug
```

## Plan/evaluate

```bash
python -m mini_tt.plan \
  --checkpoint runs/halfcheetah_small/checkpoint_latest.pt \
  --download \
  --episodes 1 \
  --horizon 5 \
  --beam_width 16 \
  --topk 8 \
  --device auto \
  --verbose
```

## Scaling knobs

Smaller/faster:

- `--max_episodes 5` to `50`
- `--n_layer 1` or `2`
- `--n_head 1` or `2`
- `--n_embd 32` or `64`
- `--n_bins 16` or `32`
- `--horizon 3`
- `--beam_width 4` to `16`

Closer to the original repo:

- `--max_episodes 1000` or omit it
- `--n_bins 100`
- `--n_layer 4`
- `--n_head 4`
- `--n_embd 128`
- `--horizon 5` for HalfCheetah, larger for Hopper/Walker2d
- `--beam_width 32` or higher

## Important differences from the original repo

1. Minari datasets are reproductions/modern datasets, not byte-for-byte D4RL HDF5 files.
2. Gymnasium separates `terminated` from `truncated`; this code treats either one as an episode boundary during evaluation, but preserves the distinction in the step call.
3. The compact planner is intentionally simple. It uses beam search over discretized tokens and ranks candidates by predicted discounted reward/value plus a small log-probability bonus.
4. This code assumes continuous Box observations/actions, matching MuJoCo locomotion tasks.

## 1 GB HalfCheetah profile

For a 1 GB target, use the commands in `LOW_MEMORY_HALFCHEETAH.md`, or run:

```bash
bash run_halfcheetah_1gb_train.sh
bash run_halfcheetah_1gb_plan.sh
```

This profile uses HalfCheetah, Minari's `mujoco/halfcheetah/medium-v0`, Gymnasium,
training plus beam-search planning, and very small model/data settings.
