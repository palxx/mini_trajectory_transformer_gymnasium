#!/usr/bin/env bash
set -euo pipefail
python -m mini_tt.plan \
  --checkpoint runs/halfcheetah_1gb/checkpoint_latest.pt \
  --download \
  --episodes 1 \
  --max_steps 50 \
  --horizon 3 \
  --beam_width 4 \
  --topk 4 \
  --context_transitions 0 \
  --device cpu \
  --verbose
