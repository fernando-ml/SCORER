import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent with SCORER")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment name (default: CartPole-v1)")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds to run (default: 10)")
    parser.add_argument("--seed", type=int, default= 8,
                        help="Base random seed (default: 8)")
    parser.add_argument("--total_timesteps", type=float, default=1e6,
                        help="Total timesteps for training (default: 1e8)")
    parser.add_argument("--use_be_variance", action="store_true",
                        help="Use BE variance in perception loss")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["tanh", "relu"])
    parser.add_argument("--follower_convergence_steps", type=int, default=1,
                        help="Number of gradient steps for follower to converge to best response")
    return parser.parse_args()

import jax
import jax.numpy as jnp
import chex
import flax
import optax
import wandb
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import flashbax as fbx
import gymnax
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from utils.utils import L2Norm, L1Norm

class PerceptionNetwork(nn.Module):
    latent_dim: int
    activation: str = "tanh"
    is_minatar: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.is_minatar:
            # MinAtar architecture
            x = L1Norm()(x)
            encoded = nn.Dense(self.latent_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                            bias_init=nn.initializers.constant(0.0))(x)
            encoded = activation(encoded)
            encoded = nn.Dense(self.latent_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                            bias_init=nn.initializers.constant(0.0))(encoded)
            encoded = activation(encoded)
            encoded = L1Norm()(encoded)
            return encoded
        else:
            # Control architecture
            # x = L1Norm()(x)
            x = nn.Dense(self.latent_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        bias_init=nn.initializers.constant(0.0))(x)
            x = activation(x)
            x = L1Norm()(x)
            return x

class QNetwork(nn.Module):
    action_dim: int
    activation: str = "tanh"
    is_minatar: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.is_minatar:
            # MinAtar architecture
            x = nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        bias_init=nn.initializers.constant(0.0))(x)
            x = activation(x)
            x = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01),
                        bias_init=nn.initializers.constant(0.0))(x)
            return x
        else:
            # Control architecture
            x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        bias_init=nn.initializers.constant(0.0))(x)
            x = activation(x)
            q_values = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(x)
            return q_values

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class PerceptionTrainState(TrainState):
    target_params: flax.core.FrozenDict

class ActionTrainState(TrainState):
    target_params: flax.core.FrozenDict

def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )

    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):
        seed_value = jax.random.key_data(rng)[0]

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        rng, _rng_buffer = jax.random.split(rng)
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )

        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        _action = basic_env.action_space().sample(_rng_buffer)
        _, _env_state = env.reset(_rng_buffer, env_params)
        _obs, _, _reward, _done, _ = env.step(_rng_buffer, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        is_minatar = "minatar" in config["ENV_NAME"].lower()

        perception_network = PerceptionNetwork(
            latent_dim=config["LATENT_DIM"],
            activation=config["ACTIVATION"],
            is_minatar=is_minatar
        )

        q_network = QNetwork(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            is_minatar=is_minatar
        )

        rng, _rng1, _rng2 = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.observation_space(env_params).shape)

        perception_params = perception_network.init(_rng1, init_x)

        encoded_dummy = perception_network.apply(perception_params, init_x)
        q_params = q_network.init(_rng2, encoded_dummy)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        def perception_linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["PERCEPTION_LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        perception_lr = perception_linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["PERCEPTION_LR"]

        def create_perception_tx():
            if config.get("PERCEPTION_MAX_GRAD_NORM", 0) > 0:
                return optax.chain(
                    optax.clip_by_global_norm(config["PERCEPTION_MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=perception_lr)
                )
            else:
                return optax.adam(learning_rate=perception_lr)

        def create_q_tx():
            transforms = []
            if config.get("MAX_GRAD_NORM", 0) > 0:
                transforms.append(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]))
            transforms.append(optax.adam(learning_rate=lr))
            return optax.chain(*transforms)


        perception_tx = create_perception_tx()
        q_tx = create_q_tx()

        perception_state = PerceptionTrainState.create(
            apply_fn=perception_network.apply,
            params=perception_params,
            target_params=jax.tree.map(lambda x: jnp.copy(x), perception_params),
            tx=perception_tx
        )

        action_state = ActionTrainState.create(
            apply_fn=q_network.apply,
            params=q_params,
            target_params=jax.tree.map(lambda x: jnp.copy(x), q_params),
            tx=q_tx
        )

        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(rng, 2)
            eps = jnp.clip(
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
                config["EPSILON_START"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)
            chosen_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape) < eps,
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),
                greedy_actions,
            )
            return chosen_actions

        def _update_step(runner_state, unused):
            perception_state, action_state, buffer_state, env_state, last_obs, timesteps, rng = runner_state

            rng, rng_a, rng_s = jax.random.split(rng, 3)

            encoded_state = perception_network.apply(perception_state.params, last_obs)

            q_vals = q_network.apply(action_state.params, encoded_state)
            action = eps_greedy_exploration(rng_a, q_vals, timesteps)

            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )

            timesteps = timesteps + config["NUM_ENVS"]

            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            def _learn_leader(perception_state, action_state, rng):
                """Leader (Q-network) update - happens less frequently"""
                learn_batch = buffer.sample(buffer_state, rng).experience

                encoded_state = perception_network.apply(perception_state.params, learn_batch.first.obs)

                encoded_next_state = perception_network.apply(
                    perception_state.target_params, learn_batch.second.obs
                )
                q_next_target = q_network.apply(
                    action_state.target_params, encoded_next_state
                )
                next_values = jnp.max(q_next_target, axis=-1)

                td_target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * next_values
                )
                td_target = jax.lax.stop_gradient(td_target)

                q_current = q_network.apply(action_state.params, encoded_state)
                q_selected = jnp.take_along_axis(
                    q_current,
                    jnp.expand_dims(learn_batch.first.action, axis=-1),
                    axis=-1
                ).squeeze(axis=-1)

                def q_loss_fn(q_params):
                    # Q-network (leader) loss function
                    q_vals = q_network.apply(q_params, jax.lax.stop_gradient(encoded_state))
                    q_selected = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1
                    ).squeeze(axis=-1)

                    td_errors = jax.lax.stop_gradient(td_target) - q_selected

                    loss = jnp.mean(jnp.square(td_errors))
                    return loss, {"td_error_mean": jnp.mean(td_errors),
                                 "td_error_std": jnp.std(td_errors)}

                (q_loss, q_metrics), q_grads = jax.value_and_grad(
                    q_loss_fn, has_aux=True
                )(action_state.params)

                action_state = action_state.apply_gradients(grads=q_grads)

                return action_state, q_loss, q_metrics

            def _learn_follower(perception_state, action_state, rng):
                """Follower (Perception) computes best response to leader's strategy"""

                def follower_step(carry, _):
                    perception_state, rng = carry
                    rng, rng_sample = jax.random.split(rng)

                    learn_batch = buffer.sample(buffer_state, rng_sample).experience

                    # Compute TD target using current perception
                    encoded_next_state = perception_network.apply(
                        perception_state.target_params, learn_batch.second.obs
                    )
                    q_next_target = q_network.apply(
                        action_state.target_params, encoded_next_state
                    )
                    next_values = jnp.max(q_next_target, axis=-1)
                    td_target = (
                        learn_batch.first.reward
                        + (1 - learn_batch.first.done) * config["GAMMA"] * next_values
                    )
                    td_target = jax.lax.stop_gradient(td_target)

                    def perception_loss_fn(perception_params):
                        encoded_state_new = perception_network.apply(
                            perception_params, learn_batch.first.obs
                        )

                        # Use FIXED Q-network params - leader has committed to this strategy
                        q_vals = q_network.apply(
                            jax.lax.stop_gradient(action_state.params),  # Fixed leader strategy
                            encoded_state_new
                        )

                        q_selected = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(learn_batch.first.action, axis=-1),
                            axis=-1
                        ).squeeze(axis=-1)

                        td_errors = jax.lax.stop_gradient(td_target) - q_selected

                        # All loss components
                        msbe = jnp.mean(jnp.square(td_errors))
                        td_mean = jnp.mean(td_errors)
                        be_variance = jnp.mean(jnp.square(td_errors - td_mean))

                        # MSBE as fallback
                        loss = msbe * (1.0 - config["USE_BE_VARIANCE"])

                        if config["USE_BE_VARIANCE"]:
                            loss = loss + be_variance


                        return loss, {
                            "be_variance": be_variance,
                            "msbe": msbe
                        }

                    (perception_loss, perception_metrics), perception_grads = jax.value_and_grad(
                        perception_loss_fn, has_aux=True
                    )(perception_state.params)

                    perception_state = perception_state.apply_gradients(grads=perception_grads)

                    return (perception_state, rng), perception_metrics

                (perception_state, _), perception_metrics = jax.lax.scan(
                    follower_step,
                    (perception_state, rng),
                    None,
                    config["FOLLOWER_CONVERGENCE_STEPS"]
                )

                # Return the last metrics from convergence
                perception_metrics = jax.tree.map(lambda x: x[-1], perception_metrics)

                return perception_state, perception_metrics

            # Stackelberg game sequential updates
            rng, rng_leader, rng_follower = jax.random.split(rng, 3)

            # Check if buffer can sample
            can_learn = buffer.can_sample(buffer_state) & (timesteps > config["LEARNING_STARTS"])

            # STEP 1: Leader (Q-network) update
            is_leader_update_time = can_learn & (timesteps % config["LEADER_UPDATE_INTERVAL"] == 0)

            def do_leader_update():
                new_action_state, loss, metrics = _learn_leader(perception_state, action_state, rng_leader)
                return new_action_state, loss, metrics

            def skip_leader_update():
                return action_state, jnp.array(0.0), {"td_error_mean": jnp.array(0.0),
                                                       "td_error_std": jnp.array(0.0)}

            action_state, q_loss, q_metrics = jax.lax.cond(
                is_leader_update_time,
                lambda: do_leader_update(),
                lambda: skip_leader_update()
            )

            # STEP 2: Follower (Perception) best response
            is_follower_update_time = can_learn & (timesteps % config["FOLLOWER_UPDATE_INTERVAL"] == 0)

            def do_follower_update():
                return _learn_follower(perception_state, action_state, rng_follower)

            def skip_follower_update():
                return perception_state, {"be_variance": jnp.array(0.0),
                                         "msbe": jnp.array(0.0)}

            perception_state, perception_metrics = jax.lax.cond(
                is_follower_update_time,
                lambda: do_follower_update(),
                lambda: skip_follower_update()
            )

            # Update target networks
            def update_perception_target():
                return perception_state.replace(
                    target_params=optax.incremental_update(
                        perception_state.params,
                        perception_state.target_params,
                        config["TAU"]
                    )
                )

            def skip_perception_target_update():
                return perception_state

            perception_state = jax.lax.cond(
                timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda: update_perception_target(),
                lambda: skip_perception_target_update()
            )

            def update_action_target():
                return action_state.replace(
                    target_params=optax.incremental_update(
                        action_state.params,
                        action_state.target_params,
                        config["TAU"]
                    )
                )

            def skip_action_target_update():
                return action_state

            action_state = jax.lax.cond(
                timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda: update_action_target(),
                lambda: skip_action_target_update()
            )

            # Combine metrics
            metrics = {
                **q_metrics,
                **perception_metrics,
                "q_loss": q_loss,
                "perception_loss": perception_metrics.get("msbe", 0.0),
                "is_leader_update": jnp.where(is_leader_update_time, jnp.array(1.0), jnp.array(0.0)),
                "is_follower_update": jnp.where(is_follower_update_time, jnp.array(1.0), jnp.array(0.0))
            }

            full_metrics = {
                "timesteps": timesteps,
                "returns": info["returned_episode_returns"].mean(),
                **metrics
            }

            runner_state = (
                perception_state,
                action_state,
                buffer_state,
                env_state,
                obs,
                timesteps,
                rng,
            )
            return runner_state, full_metrics

        timesteps = 0

        rng, _rng = jax.random.split(rng)
        runner_state = (
            perception_state,
            action_state,
            buffer_state,
            env_state,
            init_obs,
            timesteps,
            _rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

def run_single_seed(config, seed):
    if config.get("WANDB_MODE", "disabled") == "online":
        wandb.init(
            project=config.get("PROJECT", "SCORER_DQN_Jax"),
            tags=["SCORER_DQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'SCORER_DQN_{config["ENV_NAME"]}_{seed}',
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

    filename_prefix = f"SCORER_DQN_{config['ENV_NAME']}_{loss_str}_{seed}"
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
    os.makedirs("results/DQN", exist_ok=True)

    all_returns = outs["metrics"]["returns"]
    all_timesteps = outs["metrics"]["timesteps"][0]

    # Downsample data to reduce file size - save every 1000 updates
    downsample_factor = 1000
    indices = jnp.arange(0, len(all_timesteps), downsample_factor)
    all_returns = all_returns[:, indices]
    all_timesteps = all_timesteps[indices]

    loss_components = []
    if config["USE_MSBE"]:
        loss_components.append("MSBE")
    if config["USE_BE_VARIANCE"]:
        loss_components.append("BEVar")
    loss_str = '+'.join(loss_components) if loss_components else "MSBE"

    filename_prefix = f"DQN/{config['ENV_NAME']}_{loss_str}_{config['SEED']}"

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
        plt.title(f"SCORER DQN: {config['ENV_NAME']} with {loss_str}")
        plt.xlabel("Timesteps")
        plt.ylabel("Returns")
        plt.savefig(f"results/{filename_prefix}_plot.png")
        plt.close()
    except:
        print("Could not create plot")

def set_environment_defaults(config, env_name):
    is_minatar = "minatar" in env_name.lower()

    if is_minatar:
        # MinAtar defaults during our experiments
        config.update({
            "NUM_ENVS": 128,
            "BUFFER_SIZE": 100_000,
            "BUFFER_BATCH_SIZE": 64,
            "EPSILON_START": 1.0,
            "EPSILON_FINISH": 0.01,
            "EPSILON_ANNEAL_TIME": 250_000,
            "TARGET_UPDATE_INTERVAL": 1e3,
            "LR": 1e-4,
            "LEADER_UPDATE_INTERVAL": 4,
            "FOLLOWER_UPDATE_INTERVAL": 4,
            "LATENT_DIM": 128,
            "PERCEPTION_LR": 5e-4,  # Follower learns faster to converge to best response
        })
    else:
        # Control environments defaults during our experiments
        config.update({
            "NUM_ENVS": 10,
            "BUFFER_SIZE": 50_000,
            "BUFFER_BATCH_SIZE": 64,
            "EPSILON_START": 1.0,
            "EPSILON_FINISH": 0.05,
            "EPSILON_ANNEAL_TIME": 50_000,
            "TARGET_UPDATE_INTERVAL": 1000,
            "LR": 1e-4,
            "LEARNING_STARTS": 1000,
            "LEADER_UPDATE_INTERVAL": 10,
            "FOLLOWER_UPDATE_INTERVAL": 10,
            "LATENT_DIM": 64,
            "PERCEPTION_LR": 3e-4,  # Follower learns faster to converge to best response
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
        "TAU": 1.0,
        "ENV_NAME": args.env,
        "SEED": args.seed,
        "ACTIVATION": args.activation,
        "DEBUG": False,
        "WANDB_MODE": "online" if args.num_seeds == 1 else "disabled",
        "PROJECT": "SCORER_DQN",
        "MAX_GRAD_NORM": 0.5,
        "PERCEPTION_MAX_GRAD_NORM": 0.5,
        "LEADER_UPDATE_INTERVAL": 4,
        "FOLLOWER_UPDATE_INTERVAL": 4, # both update every 4 steps as in TTSA
        "FOLLOWER_CONVERGENCE_STEPS": args.follower_convergence_steps,  # following TTSA we stick to 1 step
        "LR_LINEAR_DECAY": True,
        "USE_MSBE": not (args.use_be_variance),  # Default to MSBE if no other loss specified
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