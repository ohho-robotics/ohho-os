# PPO recipe

`family: ppo` trains an on-policy policy in simulation and writes an ONNX actor. The recipe is `ohho.recipe/v1`. Robot-specific environments plug in as tasks. The recipe file does not change when a new robot is added.

```bash
ohho train ohho/train/examples/ppo.yaml --set output.dir=/tmp/ppo-cartpole
```

The shipped example trains CartPole on CPU. It does not read the demonstration dataset named in `dataset.path`. That path is stored on the run manifest. A path that is not on disk does not stop the run. `dataset.version_id` alone exits `4`, the same as the other families.

`precision` must be `fp32` in this build. `device` is `cpu`, `cuda`, `mps`, or `auto`. `cuda` and `mps` are refused when that device is not present.

## Parameters

| Hyperparameter | Default | Meaning |
| --- | --- | --- |
| `task` | `cartpole` | Task-registry id. |
| `backend` | `builtin` | `builtin`, `rsl_rl`, `skrl`, or `isaac`. |
| `num_envs` | `4` | Environments stepped together. |
| `lr` | `0.0003` | Adam learning rate. |
| `clip_coef` | `0.2` | PPO clip range, in `(0, 1]`. |
| `entropy_coef` | `0` | Entropy bonus. The example uses `0.01`. |
| `gae_lambda` | `0.95` | GAE λ. |
| `gamma` | `0.99` | Discount. |
| `rollout_steps` | `32` | Steps collected per environment before an update. |
| `reward_weights` | `{}` | Passed through to the task. |
| `updates` | `8` | Rollout updates. `train_steps` is used when `updates` is omitted. |
| `num_epochs` | `4` | Passes over the rollout per update. |
| `batch_size` | `64` | Minibatch size. A size that does not divide the rollout becomes one batch on the external backends. |
| `hidden_size` | `32` | Width of the built-in actor and critic. |
| `episode_steps` | task default | Horizon. Omit it to keep the task's reference horizon. |
| `seed` | unset | Seeds Python, NumPy, and torch when set, and the task's own RNGs. |

`reward_weights` for the built-in tasks:

| Key | Default | Effect |
| --- | --- | --- |
| `alive` | `1` | Added on every step, including the step that ends the episode. |
| `x_penalty` | `0` | Subtracted times cart position squared. |
| `theta_penalty` | `0` | Subtracted times pole angle squared. |

A plugin task may read other names from the same mapping. Unknown recipe keys are rejected.

## Tasks

Register an object with `id`, `available()`, `unavailable_reason()`, `observation_dim()`, `action_dim()`, `discrete()`, and `make(num_envs, reward_weights, seed, episode_steps)`. `requires_gpu()` returns true when this build must refuse the task. An installed distribution can use the `ohho.rl_tasks` entry point. A second implementation of an id that is already registered is refused.

```python
from ohho.train.rl_tasks import register

class MyTask:
    id = "my_robot_walk"

    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def requires_gpu(self) -> bool:
        return False

    def observation_dim(self) -> int:
        return 8

    def action_dim(self) -> int:
        return 2

    def discrete(self) -> bool:
        return False

    def make(self, *, num_envs, reward_weights, seed, episode_steps=None):
        raise NotImplementedError

register(MyTask(), source="my_package")
```

```toml
[project.entry-points."ohho.rl_tasks"]
my_robot_walk = "my_package.tasks:TASK"
```

Shipped tasks:

| Id | Where it runs | Solved bar |
| --- | --- | --- |
| `cartpole` | CPU. No extra packages. Discrete actions. Horizon 500. | Mean episode return **475** (CartPole-v1). |
| `mujoco-inverted-pendulum` | CPU when `mujoco` is installed (`ohho-os[humanoid-sim]`). Continuous cart force. Horizon 1000. | Mean episode return **950**. |
| `isaac-velocity` | Not in this build. | GPU follow-up. |

`cartpole` is the classic cart-pole: state `(x, x_dot, theta, theta_dot)`, action `0` or `1`. An episode succeeds when it reaches the horizon inside the position and angle limits.

## Reference MuJoCo task

The reference task is `mujoco-inverted-pendulum`.

- Model: a cart on a slide joint and a pole on a hinge, timestep 0.02 s, no contact. The MJCF ships in `ohho.train.rl_tasks` and does not download meshes.
- Observation: `(x, theta, x_dot, theta_dot)`.
- Action: one force in `[-1, 1]` after clipping.
- Episode ends when `|theta| > 0.2` radians.
- Default reward weights pay `1` per step, so the return equals the number of steps survived, including the failing step. That is the same scoring as Gymnasium `InvertedPendulum-v4`.
- **Reward threshold: 950.** That is the mean episode return over evaluation episodes on the 1000-step horizon. This build does not run that evaluation to the threshold. A long training run that reaches it, and the Isaac Lab GPU run, are follow-up.

Select it with `hyperparameters.task=mujoco-inverted-pendulum`. The built-in PPO can train it on CPU for a smoke run. Reaching 950 is not what the shipped example does.

## Backends

| `backend` | Behaviour |
| --- | --- |
| `builtin` | Pure PyTorch PPO. This is the CPU path the tests run. |
| `rsl_rl` | Optional import of `rsl_rl` 2.3 `OnPolicyRunner`. Continuous tasks only. The recipe maps `lr`, `clip_coef`, `entropy_coef`, `gae_lambda`, `gamma`, `rollout_steps`, and the env count into that runner's config. |
| `skrl` | Optional import of skrl 1.4 `PPO` and `SequentialTrainer`. Continuous tasks only. The same parameters are written into `PPO_DEFAULT_CONFIG`. |
| `isaac` | Refused. Isaac Lab PPO needs a GPU and Isaac Lab. GPU training is a follow-up in this build. |

A missing `rsl_rl` or `skrl` install fails before training and names the package. It does not fall through to a fake result. A discrete task on those backends is refused; use `backend: builtin`.

`isaac-velocity` is refused the same way, whichever backend is selected. The message says the task needs a GPU and Isaac Lab and that GPU training is a follow-up.

## Metrics

Each update emits a progress line on the existing train stream:

```json
{"stream": "ohho.train.progress/v1", "event": "metrics", "family": "ppo", "update": 1, "episode_reward": 12.5, "success": 0.0}
```

`episode_reward` is the mean return of episodes that finished in that update. When none finished, it is the mean of the returns still in progress, and `success` is `0`. `success` is the fraction of finished episodes that reached the horizon without a failure, in `[0, 1]`.

The same rows are stored in `metrics.json` under `intermediate`, with `policy_loss` added when that update computed one. The file also has the last `episode_reward`, the last `success`, and `final_loss`. `final_loss` is the mean clipped surrogate loss of the last update for the built-in trainer, and `0` when an external backend does not report a loss. `ohho train` reads `final_loss` from that file and records `metrics_path` on the run manifest, the same path ACT uses. Eval hooks are still only recorded. The trainer reports reward and success itself.

## ONNX actor

The run writes:

| File | Contents |
| --- | --- |
| `actor.pt` | Checkpoint. `kind` is `policy_actor` for the built-in network. |
| `actor.onnx` | Deterministic actor. Input `observation` is `(batch, observation_dim)`, float32. Output `action` is logits (discrete) or the Gaussian mean (continuous). |
| `config.json` | Task, backend, sizes, reward threshold, resolved hyperparameters. |
| `metrics.json` | `final_loss`, last episode reward and success, and `intermediate`. |
| `run_manifest.json` | `ohho.run/v1`, written by `ohho train`, not by the family. |

Export needs `torch` and `onnx` (`pip install 'ohho-os[train]'` and `pip install 'ohho-os[humanoid-sim]'`, or the equivalent packages). The check happens before training. ONNX Runtime is what a consumer uses to load `actor.onnx`. The recipe does not import it.

`base_weights` is recorded and not loaded. `base_weights.loaded` stays false.

## GPU follow-up

This build does not train Isaac Lab, and it does not claim the MuJoCo reference task reaches a return of 950. Those runs need a GPU and a long horizon. `backend: isaac` and `task: isaac-velocity` fail immediately with that reason. TensorRT is not part of this recipe.
