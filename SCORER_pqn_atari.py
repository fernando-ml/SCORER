# built upon https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_atari.py
import os
import time
from typing import Any

import chex
import envpool
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb
from flax.training.train_state import TrainState
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import hydra
from utils.atari_wrapper import JaxLogEnvPoolWrapper
from utils.utils import L1Norm, L2Norm

class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        elif self.norm_type == "l1_norm":
            normalize = lambda x: L1Norm()(x)
        elif self.norm_type == "l2_norm":
            normalize = lambda x: L2Norm()(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return x


class PerceptionNetwork(nn.Module):
    latent_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            _ = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.latent_dim)(x)
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
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
        elif self.norm_type == "l1_norm":
            normalize = lambda x: L1Norm()(x)
        elif self.norm_type == "l2_norm":
            normalize = lambda x: L2Norm()(x)
        else:
            normalize = lambda x: x

        for layer_idx in range(self.num_layers):
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

    def make_env(num_envs):
        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            **config["ENV_KWARGS"],
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env

    total_envs = (
        (config["NUM_ENVS"] + config["TEST_ENVS"])
        if config.get("TEST_DURING_TRAINING", False)
        else config["NUM_ENVS"]
    )
    env = make_env(total_envs)
    init_obs, env_state = env.reset()

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

        original_seed = rng[0]

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
            action_dim=env.single_action_space.n,
            hidden_size=config.get("Q_HIDDEN_SIZE", 64),
            num_layers=config.get("Q_NUM_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
        )

        def create_agents(rng):
            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros((1, *env.single_observation_space.shape))

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
                optax.radam(learning_rate=perception_lr)
            )

            q_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr)
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
                last_obs, env_state, rng = carry
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

                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(n_updates))
                if config.get("TEST_DURING_TRAINING", False):
                    eps = jnp.concatenate((eps, jnp.zeros(config["TEST_ENVS"])))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            if config.get("TEST_DURING_TRAINING", False):
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )

            timesteps = timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]

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
            last_q = jnp.max(transitions.q_val[-1], axis=-1)
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
                            _, updates = encoded_state
                        else:
                            updates = {"batch_stats": perception_state.batch_stats}

                        msbe = jnp.mean(jnp.square(td_errors))
                        td_mean = jnp.mean(td_errors)
                        be_variance = jnp.mean(jnp.square(td_errors - td_mean))

                        loss = jnp.zeros((), dtype=msbe.dtype)
                        if config.get("USE_MSBE", True):
                            loss = loss + msbe
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

            if config.get("TEST_DURING_TRAINING", False):
                test_infos = jax.tree_util.tree_map(lambda x: x[:, -config["TEST_ENVS"] :], infos)
                infos = jax.tree_util.tree_map(lambda x: x[:, : -config["TEST_ENVS"]], infos)
                infos.update({"test/" + k: v for k, v in test_infos.items()})

            metrics = {
                "env_step": timesteps,
                "update_steps": n_updates,
                "env_frame": timesteps * env.observation_space.shape[0],
                "timesteps": timesteps,
                "n_updates": n_updates,
                "q_loss": q_loss.mean(),
                "perception_loss": p_loss.mean(),
                "qvals": qvals.mean(),
                "be_variance": be_var.mean(),
                "msbe": msbe.mean(),
            }
            metrics.update({k: v.mean() for k, v in infos.items()})
            if config.get("TEST_DURING_TRAINING", False):
                metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (perception_state, action_state, expl_state, timesteps, n_updates, rng)

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)

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

    returns = np.array(out["metrics"]["returned_episode_returns"])
    timesteps = np.array(out["metrics"]["timesteps"])

    if config.get("WANDB_MODE", "disabled") == "online":
        for i, (ts, ret) in enumerate(zip(timesteps, returns)):
            if i % 100 == 0:
                wandb.log({
                    "returned_episode_returns": ret,
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

    all_returns = outs["metrics"]["returned_episode_returns"]
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


def run_experiment(config):
    print("Available devices:", jax.devices())
    if config["NUM_SEEDS"] == 1:
        print("Running single seed training...")
        return run_single_seed(config, config["SEED"])
    print(f"Running training with {config['NUM_SEEDS']} seeds in parallel...")
    outs = run_multiple_seeds(config)
    print("Multi-seed training complete!")
    return outs


@hydra.main(version_base=None, config_path="./atari_config", config_name="config")
def main(cfg):
    if OmegaConf.select(cfg, "HYP_TUNE"):
        raise NotImplementedError("Hyperparameter tuning not supported for SCORER PQN yet.")

    if OmegaConf.select(cfg, "alg.ALG_NAME") != "SCORER_pqn":
        cfg.alg = OmegaConf.load(to_absolute_path("atari_config/alg/SCORER_pqn_atari.yaml"))

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict.pop("defaults", None)
    alg_cfg = config_dict.pop("alg")
    config_dict = {**config_dict, **alg_cfg}

    if config_dict.get("WANDB_MODE") == "online" and config_dict.get("NUM_SEEDS", 1) > 1:
        config_dict["WANDB_MODE"] = "disabled"

    run_experiment(config_dict)


if __name__ == "__main__":
    main()