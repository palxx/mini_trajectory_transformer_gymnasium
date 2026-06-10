# Gym -> Gymnasium migration notes for the original repository

Use these changes when modernizing the original codebase directly.

## Environment import

```python
# old
import gym

# new
import gymnasium as gym
```

## Reset API

```python
# old
observation = env.reset()

# new
observation, info = env.reset(seed=seed)
```

For subsequent resets when you do not want to reseed:

```python
observation, info = env.reset()
```

## Step API

```python
# old
next_observation, reward, terminal, info = env.step(action)

# new
next_observation, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

Use `terminated` for natural task endings and `truncated` for time-limit endings.

## Max episode length

```python
# old, often custom env attribute
T = env.max_episode_steps

# new
T = env.spec.max_episode_steps if env.spec is not None else 1000
```

## D4RL dataset loading replacement

The old code calls D4RL helpers such as `env.get_dataset()` or a D4RL q-learning dataset wrapper.
A modern Gymnasium-compatible path is Minari:

```python
import minari

dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0", download=True)
env = dataset.recover_environment()

for episode in dataset.iterate_episodes():
    observations = episode.observations
    actions = episode.actions
    rewards = episode.rewards
    terminations = episode.terminations
    truncations = episode.truncations
```

## Normalized score

```python
# old D4RL style
score = env.get_normalized_score(total_reward)

# new Minari style
import numpy as np
import minari
score = minari.get_normalized_score(dataset, np.asarray([total_reward]))[0]
```
