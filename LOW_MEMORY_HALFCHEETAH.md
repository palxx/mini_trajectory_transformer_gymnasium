# Low-memory HalfCheetah run: 1 GB target

This profile is for a very small proof-of-concept run. It keeps the same components
as the full Mini Trajectory Transformer, but aggressively limits dataset size,
sequence length, model width, batch size, and beam-search width.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you are on a CPU-only machine, install the CPU build of PyTorch first from the
official PyTorch instructions, then run `pip install -e .`.

## 2. Smoke test: environment + dataset

```bash
python - <<'PY'
import minari

dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0", download=True)
env = dataset.recover_environment()
obs, info = env.reset(seed=0)
print("obs shape:", obs.shape)
print("action shape:", env.action_space.shape)
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
print("step ok:", float(reward), terminated, truncated)
env.close()
PY
```

## 3. Train: smallest practical config

```bash
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
```

Use `--device cpu` for the safest 1 GB run. If you mean 1 GB of GPU VRAM and have
more system RAM available, try `--device cuda` only after the CPU smoke test works.

## 4. Beam-search evaluation: smallest practical config

```bash
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
```

## 5. Slightly larger, still small

After the above works, increase one knob at a time:

```bash
python -m mini_tt.train \
  --dataset mujoco/halfcheetah/medium-v0 \
  --download \
  --max_episodes 10 \
  --max_steps_per_episode 500 \
  --sequence_length 8 \
  --stride 4 \
  --n_bins 32 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 48 \
  --batch_size 16 \
  --epochs 3 \
  --max_train_steps 1000 \
  --num_workers 0 \
  --device cpu \
  --out_dir runs/halfcheetah_small_plus
```

Beam search for that checkpoint:

```bash
python -m mini_tt.plan \
  --checkpoint runs/halfcheetah_small_plus/checkpoint_latest.pt \
  --download \
  --episodes 1 \
  --max_steps 100 \
  --horizon 5 \
  --beam_width 8 \
  --topk 4 \
  --context_transitions 0 \
  --device cpu
```

## Notes

- This is a proof-of-concept configuration, not a performance configuration.
- Good HalfCheetah returns require much larger datasets, longer training, and larger beam widths.
- For memory pressure, reduce in this order: `batch_size`, `max_episodes`, `sequence_length`, `n_embd`, `beam_width`.
