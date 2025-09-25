import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train PQN agent with SCORER")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment name (default: CartPole-v1)")
    parser.add_argument("--num_seeds", type=int, default=30,
                        help="Number of seeds to run (default: 30)")
    parser.add_argument("--seed", type=int, default=8,
                        help="Base random seed (default: 8)")
    parser.add_argument("--total_timesteps", type=float, default=1e6,
                        help="Total timesteps for training (default: 1e6)")
    parser.add_argument("--use_be_variance", action="store_true",
                        help="Use Variance in perception loss")
    return parser.parse_args()

import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax
import wandb
import matplotlib.pyplot as plt
import pandas as pd


class PerceptionNetwork(nn.Module):
    latent_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(self.latent_dim)(x)
        x = normalize(x)
        x = nn.relu(x)
        x = normalize(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    num_layers: int = 1
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class PerceptionTrainState(TrainState):
    batch_stats: Any

class ActionTrainState(TrainState):
    batch_stats: Any


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),
            greedy_actions,
        )
        return chosed_actions

    def train(rng):

        original_rng = rng[0] if hasattr(rng, '__getitem__') else jax.random.key_data(rng)[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        perception_network = PerceptionNetwork(
            latent_dim=config["LATENT_DIM"],
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        q_network = QNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("Q_HIDDEN_SIZE", 64),
            num_layers=config.get("Q_NUM_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
        )

        def create_agents(rng):
            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))

            perception_variables = perception_network.init(_rng1, init_x, train=False)
            encoded_dummy = perception_network.apply(perception_variables, init_x, train=False)
            q_variables = q_network.init(_rng2, encoded_dummy, train=False)

            # Ensure batch_stats exist
            if "batch_stats" not in perception_variables:
                perception_variables = {"params": perception_variables["params"], "batch_stats": {}}
            if "batch_stats" not in q_variables:
                q_variables = {"params": q_variables["params"], "batch_stats": {}}

            perception_lr_scheduler = optax.linear_schedule(
                init_value=config["PERCEPTION_LR"],
                end_value=1e-20,
                transition_steps=(config["NUM_UPDATES_DECAY"])
                * config["NUM_MINIBATCHES"]
                * config["NUM_EPOCHS"],
            )
            perception_lr = perception_lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["PERCEPTION_LR"]

            perception_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=perception_lr)
            )

            q_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr)
            )

            perception_state = PerceptionTrainState.create(
                apply_fn=perception_network.apply,
                params=perception_variables["params"],
                batch_stats=perception_variables["batch_stats"],
                tx=perception_tx
            )

            action_state = ActionTrainState.create(
                apply_fn=q_network.apply,
                params=q_variables["params"],
                batch_stats=q_variables["batch_stats"],
                tx=q_tx
            )

            return perception_state, action_state

        rng, _rng = jax.random.split(rng)
        perception_state, action_state = create_agents(rng)

        def _update_step(runner_state, unused):
            perception_state, action_state, expl_state, timesteps, n_updates, rng = runner_state

            def _step_env(carry, _):
                last_obs, env_state, current_timesteps, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # Division of labor: perception encodes, Q-network selects actions
                encoded_state = perception_network.apply(
                    {"params": perception_state.params, "batch_stats": perception_state.batch_stats},
                    last_obs,
                    train=False
                )

                q_vals = q_network.apply(
                    {"params": action_state.params, "batch_stats": action_state.batch_stats},
                    encoded_state,
                    train=False
                )

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                current_timesteps = current_timesteps + config["NUM_ENVS"]

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, current_timesteps, rng), (transition, info, current_timesteps)

            rng, _rng = jax.random.split(rng)
            (*expl_state, timesteps, rng), (transitions, infos, step_timesteps) = jax.lax.scan(
                _step_env,
                (*expl_state, timesteps, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            last_encoded = perception_network.apply(
                {"params": perception_state.params, "batch_stats": perception_state.batch_stats},
                transitions.next_obs[-1],
                train=False
            )
            last_q = q_network.apply(
                {"params": action_state.params, "batch_stats": action_state.batch_stats},
                last_encoded,
                train=False
            )
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward

                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # LEARN FROM TRAJECTORIES (both networks update together, like PQN)
            def _learn_epoch(carry, _):
                perception_state, action_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):
                    perception_state, action_state, rng = carry
                    minibatch, target = minibatch_and_target

                    # Q-NETWORK UPDATE (with fixed perception)
                    def q_loss_fn(q_params):
                        # Use stop_gradient on perception encoding (leader commits first)
                        encoded_state = perception_network.apply(
                            {"params": perception_state.params, "batch_stats": perception_state.batch_stats},
                            minibatch.obs,
                            train=False
                        )

                        q_vals, updates = q_network.apply(
                            {"params": q_params, "batch_stats": action_state.batch_stats},
                            jax.lax.stop_gradient(encoded_state),  # Stop gradient from perception
                            train=True,
                            mutable=["batch_stats"],
                        )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                        return loss, (updates, chosen_action_qvals)

                    (q_loss, (q_updates, qvals)), q_grads = jax.value_and_grad(
                        q_loss_fn, has_aux=True
                    )(action_state.params)

                    action_state = action_state.apply_gradients(grads=q_grads)
                    action_state = action_state.replace(batch_stats=q_updates["batch_stats"])

                    # FOLLOWER UPDATE (Perception with fixed Q-network)
                    def perception_loss_fn(perception_params):
                        encoded_state = perception_network.apply(
                            {"params": perception_params, "batch_stats": perception_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )

                        # Use fixed Q-network params (follower responds to leader)
                        q_vals = q_network.apply(
                            {"params": jax.lax.stop_gradient(action_state.params),
                             "batch_stats": action_state.batch_stats},
                            encoded_state[0] if isinstance(encoded_state, tuple) else encoded_state,
                            train=False
                        )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        td_errors = jax.lax.stop_gradient(target) - chosen_action_qvals

                        if isinstance(encoded_state, tuple):
                            encoded_state_new, updates = encoded_state
                        else:
                            encoded_state_new = encoded_state
                            updates = {"batch_stats": perception_state.batch_stats}

                        # Perception loss options (SCORER variants)
                        msbe = jnp.mean(jnp.square(td_errors))
                        td_mean = jnp.mean(td_errors)
                        be_variance = jnp.mean(jnp.square(td_errors - td_mean))

                        # Default to MSBE
                        loss = msbe * (1.0 - config["USE_BE_VARIANCE"])

                        if config["USE_BE_VARIANCE"]:
                            loss = loss + be_variance

                        return loss, (updates, be_variance, msbe)

                    (p_loss, (p_updates, be_var, msbe)), p_grads = jax.value_and_grad(
                        perception_loss_fn, has_aux=True
                    )(perception_state.params)

                    perception_state = perception_state.apply_gradients(grads=p_grads)
                    perception_state = perception_state.replace(batch_stats=p_updates["batch_stats"])

                    return (perception_state, action_state, rng), (q_loss, p_loss, qvals, be_var, msbe)

                def preprocess_transition(x, rng):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng, x)
                    x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (perception_state, action_state, rng), (q_loss, p_loss, qvals, be_var, msbe) = jax.lax.scan(
                    _learn_phase, (perception_state, action_state, rng), (minibatches, targets)
                )

                return (perception_state, action_state, rng), (q_loss, p_loss, qvals, be_var, msbe)

            # Run learning epochs
            rng, _rng = jax.random.split(rng)
            (perception_state, action_state, rng), (q_loss, p_loss, qvals, be_var, msbe) = jax.lax.scan(
                _learn_epoch, (perception_state, action_state, rng), None, config["NUM_EPOCHS"]
            )

            n_updates = n_updates + 1

            episode_returns = infos["returned_episode_returns"]
            mean_episode_returns = jnp.mean(episode_returns)

            metrics = {
                "timesteps": timesteps,
                "returns": mean_episode_returns,
                "n_updates": n_updates,
                "q_loss": q_loss.mean(),
                "perception_loss": p_loss.mean(),
                "qvals": qvals.mean(),
                "be_variance": be_var.mean(),
                "msbe": msbe.mean(),
            }
            metrics.update({k: v.mean() for k, v in infos.items()})

            runner_state = (perception_state, action_state, expl_state, timesteps, n_updates, rng)

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng)

        rng, _rng = jax.random.split(rng)
        timesteps = 0
        n_updates = 0
        runner_state = (perception_state, action_state, expl_state, timesteps, n_updates, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def run_single_seed(config, seed):
    if config.get("WANDB_MODE", "disabled") == "online":
        wandb.init(
            project=config.get("PROJECT", "SCORER_PQN_Jax"),
            tags=["SCORER_PQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'SCORER_PQN_{config["ENV_NAME"]}_{seed}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(seed)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    print(f"Starting training for seed {seed}...")
    out = train_jit(rng)

    returns = np.array(out["metrics"]["returns"])
    timesteps = np.array(out["metrics"]["timesteps"])

    if config.get("WANDB_MODE", "disabled") == "online":
        for i, (ts, ret) in enumerate(zip(timesteps, returns)):
            if i % 100 == 0:
                wandb.log({
                    "returns": ret,
                    "timesteps": ts,
                })
        wandb.finish()

    os.makedirs("results", exist_ok=True)

    loss_components = []
    if config["USE_MSBE"]:
        loss_components.append("MSBE")
    if config["USE_BE_VARIANCE"]:
        loss_components.append("BEVar")
    loss_str = '+'.join(loss_components) if loss_components else "MSBE"

    filename_prefix = f"SCORER_PQN_{config['ENV_NAME']}_{loss_str}_{seed}"
    np.savetxt(f"results/{filename_prefix}_returns.csv", returns, delimiter=",")
    np.savetxt(f"results/{filename_prefix}_timesteps.csv", timesteps, delimiter=",")

    return out


def run_multiple_seeds(config):
    base_seed = config["SEED"]
    num_seeds = config["NUM_SEEDS"]

    print(f"Generating {num_seeds} seeds starting from base seed {base_seed}")
    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, num_seeds)
    seed_values = [int(jax.random.key_data(k)[0]) for k in rngs]

    train_fn = make_train(config)
    train_vjit = jax.jit(jax.vmap(train_fn))

    print(f"Starting parallel training for {num_seeds} seeds...")
    t0 = time.time()
    outs = jax.block_until_ready(train_vjit(rngs))
    training_time = time.time() - t0
    print(f"Training completed in {training_time:.2f} seconds")

    process_results(config, outs, seed_values)

    return outs


def process_results(config, outs, seed_values):
    os.makedirs("results/SCORER_PQN", exist_ok=True)

    all_returns = outs["metrics"]["returns"]
    all_timesteps = outs["metrics"]["timesteps"]

    all_returns = np.array(all_returns)
    all_timesteps = np.array(all_timesteps)

    if len(all_timesteps.shape) > 1:
        all_timesteps = all_timesteps[0]

    if all_returns.size == 0:
        print("Warning: No returns data found")
        return

    loss_components = []
    if config["USE_MSBE"]:
        loss_components.append("MSBE")
    if config["USE_BE_VARIANCE"]:
        loss_components.append("BEVar")
    loss_str = '+'.join(loss_components) if loss_components else "MSBE"

    filename_prefix = f"SCORER_PQN/{config['ENV_NAME']}_{loss_str}_{config['SEED']}"

    returns_df = pd.DataFrame()
    returns_df['timesteps'] = all_timesteps

    for i, seed in enumerate(seed_values):
        returns_df[f'seed_{seed}'] = all_returns[i]

    returns_df.to_csv(f"results/{filename_prefix}_all_returns.csv", index=False)

    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)

    print(f"Results summary for {config['ENV_NAME']} ({config['NUM_SEEDS']} seeds):")
    print(f"Final mean return: {mean_returns[-1]:.2f} ± {std_returns[-1]:.2f}")
    print(f"Best seed return: {np.max(all_returns[:, -1]):.2f}")
    print(f"Worst seed return: {np.min(all_returns[:, -1]):.2f}")
    print(f"Results saved to results/{filename_prefix}_all_returns.csv")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(all_timesteps, mean_returns)
        plt.fill_between(all_timesteps,
                         mean_returns - std_returns,
                         mean_returns + std_returns,
                         alpha=0.2)
        plt.title(f"SCORER PQN: {config['ENV_NAME']} with {loss_str}")
        plt.xlabel("Timesteps")
        plt.ylabel("Returns")
        plt.savefig(f"results/{filename_prefix}_plot.png")
        plt.close()
    except:
        print("Could not create plot")


def set_environment_defaults(config, env_name):
    is_minatar = "minatar" in env_name.lower()

    if is_minatar:
        config.update({
            "NUM_ENVS": 32,
            "NUM_STEPS": 64,
            "NUM_MINIBATCHES": 8,
            "NUM_EPOCHS": 2,
            "EPS_START": 1.0,
            "EPS_FINISH": 0.01,
            "EPS_DECAY": 0.8,
            "LR": 1e-4,
            "PERCEPTION_LR": 5e-4,
            "LAMBDA": 0.95,
            "MAX_GRAD_NORM": 0.5,
            "LATENT_DIM": 128,
            "Q_HIDDEN_SIZE": 64,
            "Q_NUM_LAYERS": 1,
            "NORM_TYPE": "layer_norm",
            "NORM_INPUT": False,
            "REW_SCALE": 1,
            "TOTAL_TIMESTEPS_DECAY": 1e7,
            "LR_LINEAR_DECAY": True,
        })
    else:
        config.update({
            "NUM_ENVS": 10,
            "NUM_STEPS": 128,
            "NUM_MINIBATCHES": 4,
            "NUM_EPOCHS": 2,
            "EPS_START": 1.0,
            "EPS_FINISH": 0.05,
            "EPS_DECAY": 0.5,
            "LR": 1e-4,
            "PERCEPTION_LR": 5e-4,
            "LAMBDA": 0.95,
            "MAX_GRAD_NORM": 0.5,
            "LATENT_DIM": 64,
            "Q_HIDDEN_SIZE": 32,
            "Q_NUM_LAYERS": 1,
            "NORM_TYPE": "layer_norm",
            "NORM_INPUT": False,
            "REW_SCALE": 1,
            "TOTAL_TIMESTEPS_DECAY": 5e5,
            "LR_LINEAR_DECAY": True,
        })

    if "TOTAL_TIMESTEPS" not in config:
        config["TOTAL_TIMESTEPS"] = 1e6 if not is_minatar else 1e8

    return config


if __name__ == "__main__":
    args = parse_args()

    config = {
        "NUM_SEEDS": args.num_seeds,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "GAMMA": 0.99,
        "ENV_NAME": args.env,
        "SEED": args.seed,
        "WANDB_MODE": "online" if args.num_seeds == 1 else "disabled",
        "PROJECT": "SCORER_PQN",
        "USE_MSBE": not (args.use_be_variance),
        "USE_BE_VARIANCE": args.use_be_variance,
    }

    config = set_environment_defaults(config, args.env)

    print("Available devices:", jax.devices())

    if config["NUM_SEEDS"] == 1:
        print("Running single seed training...")
        out = run_single_seed(config, config["SEED"])
    else:
        print(f"Running training with {config['NUM_SEEDS']} seeds in parallel...")
        outs = run_multiple_seeds(config)
        print("Multi-seed training complete!")