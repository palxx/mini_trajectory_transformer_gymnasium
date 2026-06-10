#!/usr/bin/env bash
set -euo pipefail
# "Aggressive-max" config for an Apple M1 (16GB RAM, MPS, 8 cores).
# Full mujoco/halfcheetah/medium-v0 dataset (1000 episodes / 1M steps).
# ~109 min/epoch at batch_size=128 -> ~9 hours for 5 epochs.
python -m mini_tt.train \
  --dataset mujoco/halfcheetah/medium-v0 \
  --download \
  --max_episodes 1000 \
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
  --out_dir runs/halfcheetah_m1_max
