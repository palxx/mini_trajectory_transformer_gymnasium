#!/usr/bin/env bash
set -euo pipefail
# Plan with the halfcheetah_20ep checkpoint. The mujoco/halfcheetah/medium-v0
# dataset has episode returns averaging ~12,500 (range ~2,700-14,000), so we
# condition on a target return of 12,000 (Decision-Transformer style, requires
# context_transitions > 0 to take effect).
.venv/bin/python -m mini_tt.plan \
  --checkpoint runs/halfcheetah_20ep/checkpoint_latest.pt \
  --download \
  --episodes 1 \
  --max_steps 200 \
  --horizon 3 \
  --beam_width 4 \
  --topk 4 \
  --context_transitions 3 \
  --target_return 8000 \
  --device cpu \
  --record_dir runs/videos/halfcheetah_20ep \
  --verbose
