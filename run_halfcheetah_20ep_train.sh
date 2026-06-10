#!/usr/bin/env bash
set -euo pipefail
# Same "Aggressive-max" architecture as run_halfcheetah_m1_max_train.sh, but on
# only 20 episodes (full 1000 steps each) instead of the full 1000-episode dataset.
# ~19,820 windows / batch_size 128 -> ~154 steps/epoch.
# At ~1.15-1.19 it/s (M1 MPS) -> ~2-2.5 min/epoch -> ~20-25 min for 10 epochs.
.venv/bin/python -m mini_tt.train \
  --dataset mujoco/halfcheetah/medium-v0 \
  --max_episodes 20 \
  --sequence_length 10 \
  --stride 1 \
  --n_bins 100 \
  --n_layer 6 \
  --n_head 8 \
  --n_embd 256 \
  --batch_size 128 \
  --epochs 5 \
  --num_workers 0 \
  --device mps \
  --out_dir runs/halfcheetah_20ep
