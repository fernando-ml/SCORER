import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dueling DDQN agent with SCORER")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Environment name (default: CartPole-v1)")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds to run (default: 10)")
    parser.add_argument("--seed", type=int, default=8,
                        help="Base random seed (default: 8)")
    parser.add_argument("--total_timesteps", type=float, default=1e6,
                        help="Total timesteps for training (default: 1e6)")
    parser.add_argument("--use_td_variance", action="store_true",
                        help="Use TD variance in leader loss")
    parser.add_argument("--use_norm_loss", action="store_true",
                        help="Use representation norm loss in leader loss")
    parser.add_argument("--coef_representation_norm", type=float, default=0.1,
                        help="Coefficient for representation norm loss")
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

class PerceptionNetwork(nn.Module):
    latent_dim: int
    activation: str = "relu"
    is_minatar: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        if self.is_minatar:
            # MinAtar architecture
            encoded = nn.Dense(self.latent_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                            bias_init=nn.initializers.constant(0.0))(x)
            encoded = activation(encoded)
            encoded = nn.Dense(self.latent_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                            bias_init=nn.initializers.constant(0.0))(encoded)
            encoded = activation(encoded)
            return encoded
        else:
            # Control architecture
            x = nn.Dense(self.latent_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                        bias_init=nn.initializers.constant(0.0))(x)
            x = activation(x)
            return x

class DuelingQNetwork(nn.Module):
    action_dim: int
    activation: str = "relu"
    is_minatar: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        if self.is_minatar:
            # MinAtar architecture for dueling network
            value = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                            bias_init=nn.initializers.constant(0.0))(x)
            value = activation(value)
            value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(value)
            
            advantage = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                                bias_init=nn.initializers.constant(0.0))(x)
            advantage = activation(advantage)
            advantage = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01),
                                bias_init=nn.initializers.constant(0.0))(advantage)
        else:
            # Control architecture for dueling network
            value = nn.Dense(32, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                            bias_init=nn.initializers.constant(0.0))(x)
            value = activation(value)
            value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01),
                            bias_init=nn.initializers.constant(0.0))(value)
            
            advantage = nn.Dense(32, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                                bias_init=nn.initializers.constant(0.0))(x)
            advantage = activation(advantage)
            advantage = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01),
                                bias_init=nn.initializers.constant(0.0))(advantage)

        q_values = value + (advantage - advantage.mean(axis=-1, keepdims=True))
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
        
        dueling_q_network = DuelingQNetwork(
            action_dim=env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            is_minatar=is_minatar
        )
        
        rng, _rng1, _rng2 = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        
        perception_params = perception_network.init(_rng1, init_x)
        
        encoded_dummy = perception_network.apply(perception_params, init_x)
        q_params = dueling_q_network.init(_rng2, encoded_dummy)
        
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac
        
        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        
        def create_perception_tx():
            if config.get("LEADER_MAX_GRAD_NORM", 0) > 0:
                return optax.chain(
                    optax.clip_by_global_norm(config["LEADER_MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=config["LEADER_LR"])
                )
            else:
                return optax.adam(learning_rate=config["LEADER_LR"])
        
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
            target_params=jax.tree_map(lambda x: jnp.copy(x), perception_params),
            tx=perception_tx
        )
        
        action_state = ActionTrainState.create(
            apply_fn=dueling_q_network.apply,
            params=q_params,
            target_params=jax.tree_map(lambda x: jnp.copy(x), q_params),
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
            
            q_vals = dueling_q_network.apply(action_state.params, encoded_state)
            action = eps_greedy_exploration(rng_a, q_vals, timesteps)
            
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            
            timesteps = timesteps + config["NUM_ENVS"]
            
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)
            
            def _learn_phase(perception_state, action_state, timesteps, rng):
                learn_batch = buffer.sample(buffer_state, rng).experience
                action_state_before_update = action_state
                
                encoded_state = perception_network.apply(perception_state.params, learn_batch.first.obs)
                
                encoded_next_state = perception_network.apply(
                    perception_state.target_params, learn_batch.second.obs
                )
                
                q_next = dueling_q_network.apply(action_state.params, encoded_next_state)
                next_actions = jnp.argmax(q_next, axis=-1)
                
                q_next_target = dueling_q_network.apply(
                    action_state.target_params, encoded_next_state
                )
                
                q_next_target_selected = jnp.take_along_axis(
                    q_next_target,
                    jnp.expand_dims(next_actions, axis=-1),
                    axis=-1
                ).squeeze(axis=-1)
                
                td_target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target_selected
                )
                td_target = jax.lax.stop_gradient(td_target)
                
                q_current = dueling_q_network.apply(action_state.params, encoded_state)
                q_selected = jnp.take_along_axis(
                    q_current,
                    jnp.expand_dims(learn_batch.first.action, axis=-1),
                    axis=-1
                ).squeeze(axis=-1)
                
                def follower_loss_fn(q_params):
                    q_vals = dueling_q_network.apply(q_params, jax.lax.stop_gradient(encoded_state))
                    q_selected = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1
                    ).squeeze(axis=-1)
                    
                    td_errors = jax.lax.stop_gradient(td_target) - q_selected
                    
                    loss = jnp.mean(jnp.square(td_errors))
                    return loss, {"td_error_mean": jnp.mean(td_errors),
                                 "td_error_std": jnp.std(td_errors)}
                
                (follower_loss, follower_metrics), follower_grads = jax.value_and_grad(
                    follower_loss_fn, has_aux=True
                )(action_state.params)
                
                action_state = action_state.apply_gradients(grads=follower_grads)
                
                def leader_loss_fn(perception_params):
                    encoded_state_new = perception_network.apply(
                        perception_params, learn_batch.first.obs
                    )
                    
                    q_vals = dueling_q_network.apply(
                        jax.lax.stop_gradient(action_state_before_update.params),
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
                    td_variance = jnp.mean(jnp.square(td_errors - td_mean))
                    representation_norm = jnp.mean(jnp.square(encoded_state_new))
                    target_norm = 1.0
                    norm_loss = config["COEF_REPRESENTATION_NORM"] * jnp.square(target_norm - representation_norm)
                    
                    # MSBE as fallback
                    loss = msbe * (1.0 - config["USE_TD_VARIANCE"]) * (1.0 - config["USE_NORM_LOSS"]) 
                    
                    if config["USE_TD_VARIANCE"]:
                        loss = loss + td_variance
                        
                    if config["USE_NORM_LOSS"]:
                        loss = loss + norm_loss
                    
                    return loss, {
                        "td_variance": td_variance,
                        "representation_norm": representation_norm,
                        "msbe": msbe
                    }
                
                is_leader_update_time = (timesteps % config["LEADER_UPDATE_INTERVAL"] == 0)
                
                def update_leader():
                    (leader_loss, leader_metrics), leader_grads = jax.value_and_grad(
                        leader_loss_fn, has_aux=True
                    )(perception_state.params)
                    return perception_state.apply_gradients(grads=leader_grads), leader_metrics
                
                def skip_leader_update():
                    _, leader_metrics = leader_loss_fn(perception_state.params)
                    return perception_state, leader_metrics
                
                perception_state, leader_metrics = jax.lax.cond(
                    is_leader_update_time,
                    lambda: update_leader(),
                    lambda: skip_leader_update()
                )
                
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
                
                metrics = {
                    **follower_metrics,
                    **leader_metrics,
                    "follower_loss": follower_loss,
                    "leader_loss": leader_metrics.get("msbe", 0.0),
                    "is_leader_update": jnp.where(is_leader_update_time, jnp.array(1.0), jnp.array(0.0))
                }
                
                return perception_state, action_state, metrics
            
            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (timesteps > config["LEARNING_STARTS"])
                & (timesteps % config["TRAINING_INTERVAL"] == 0)
            )
            
            empty_metrics = {
                "td_error_mean": jnp.array(0.0),
                "td_error_std": jnp.array(0.0),
                "td_variance": jnp.array(0.0),
                "representation_norm": jnp.array(0.0),
                "msbe": jnp.array(0.0),
                "follower_loss": jnp.array(0.0),
                "leader_loss": jnp.array(0.0),
                "is_leader_update": jnp.array(0.0)
            }
            
            def do_learning():
                return _learn_phase(perception_state, action_state, timesteps, _rng)
            
            def skip_learning():
                return perception_state, action_state, empty_metrics
            
            perception_state, action_state, metrics = jax.lax.cond(
                is_learn_time,
                lambda: do_learning(),
                lambda: skip_learning()
            )
            
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
            project=config.get("PROJECT", "SCORER_DuelingDDQN_Jax"),
            tags=["SCORER_DuelingDDQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'SCORER_DuelingDDQN_{config["ENV_NAME"]}_{seed}',
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
    if config["USE_TD_VARIANCE"]:
        loss_components.append("TDVar")
    if config["USE_NORM_LOSS"]:
        loss_components.append("Norm")
    loss_str = '+'.join(loss_components) if loss_components else "MSBE"
    
    filename_prefix = f"SCORER_DuelingDDQN_{config['ENV_NAME']}_{loss_str}_{seed}"
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
    os.makedirs("results/DuelingDDQN", exist_ok=True)
    
    all_returns = outs["metrics"]["returns"]
    all_timesteps = outs["metrics"]["timesteps"][0]
    
    loss_components = []
    if config["USE_MSBE"]:
        loss_components.append("MSBE")
    if config["USE_TD_VARIANCE"]:
        loss_components.append("TDVar")
    if config["USE_NORM_LOSS"]:
        loss_components.append("Norm")
    loss_str = '+'.join(loss_components) if loss_components else "MSBE"
    
    filename_prefix = f"DuelingDDQN/{config['ENV_NAME']}_{loss_str}_{config['SEED']}"
    
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
        plt.title(f"SCORER DuelingDDQN: {config['ENV_NAME']} with {loss_str}")
        plt.xlabel("Timesteps")
        plt.ylabel("Returns")
        plt.savefig(f"results/{filename_prefix}_plot.png")
        plt.close()
    except:
        print("Could not create plot")

def set_environment_defaults(config, env_name):
    is_minatar = "minatar" in env_name.lower()
    
    if is_minatar:
        # MinAtar defaults
        config.update({
            "NUM_ENVS": 128,
            "BUFFER_SIZE": 100_000,
            "BUFFER_BATCH_SIZE": 64,
            "EPSILON_START": 1.0,
            "EPSILON_FINISH": 0.01,
            "EPSILON_ANNEAL_TIME": 250_000,
            "TARGET_UPDATE_INTERVAL": 1e3,
            "LR": 1e-4,
            "LEARNING_STARTS": 1e4,
            "TRAINING_INTERVAL": 4,
            "LATENT_DIM": 128,
            "LEADER_LR": 5e-4,
        })
    else:
        # Control environments defaults
        config.update({
            "NUM_ENVS": 10,
            "BUFFER_SIZE": 50_000,
            "BUFFER_BATCH_SIZE": 64,
            "EPSILON_START": 1.0,
            "EPSILON_FINISH": 0.05,
            "EPSILON_ANNEAL_TIME": 50_000,
            "TARGET_UPDATE_INTERVAL": 1000,
            "LR": 3e-4,
            "LEARNING_STARTS": 1000,
            "TRAINING_INTERVAL": 10,
            "LATENT_DIM": 64,
            "LEADER_LR": 3e-4,
        })
    
    if "TOTAL_TIMESTEPS" not in config:
        config["TOTAL_TIMESTEPS"] = 1e6 if not is_minatar else 1e8
    
    return config

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if "minatar" not in args.env.lower() else "1"
    
    config = {
        "NUM_SEEDS": args.num_seeds,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": args.env,
        "SEED": args.seed,
        "ACTIVATION": "relu",
        "DEBUG": False,
        "WANDB_MODE": "online" if args.num_seeds == 1 else "disabled",
        "PROJECT": "SCORER_DuelingDDQN",
        "MAX_GRAD_NORM": 0.0,
        "LEADER_MAX_GRAD_NORM": 0.0,
        "LEADER_UPDATE_INTERVAL": 8,
        "LR_LINEAR_DECAY": True,
        "USE_MSBE": not (args.use_td_variance or args.use_norm_loss),  # Default to MSBE if no other loss specified
        "USE_TD_VARIANCE": args.use_td_variance,
        "USE_NORM_LOSS": args.use_norm_loss,
        "COEF_REPRESENTATION_NORM": args.coef_representation_norm,
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