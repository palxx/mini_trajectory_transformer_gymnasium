#!/usr/bin/env bash
set -euo pipefail
python -m mini_tt.train \
  --dataset mujoco/halfcheetah/medium-v0 \
  --download \
  --max_episodes 3 \
  --max_steps_per_episode 300 \
  --sequence_length 5 \
  --stride 5 \
  --n_bins 16 \
  --n_layer 1 \
  --n_head 1 \
  --n_embd 32 \
  --batch_size 8 \
  --epochs 1 \
  --max_train_steps 200 \
  --num_workers 0 \
  --device cpu \
  --out_dir runs/halfcheetah_1gb
