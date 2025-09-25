import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train R2D2 agent with SCORER")
    parser.add_argument("--env", type=str, default="Navix-DoorKey-6x6-v0")
    parser.add_argument("--num_seeds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--total_timesteps", type=float, default=1e6)
    parser.add_argument("--use_be_variance", action="store_true",
                        help="Use BE variance in leader loss")
    parser.add_argument("--control_convergence_steps", type=int, default=1,
                        help="Number of gradient steps for control network (leader) to update (default: 1)")
    return parser.parse_args()

import jax
import jax.numpy as jnp
import chex
import flax
import optax
import wandb
import rlax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import flashbax as fbx
import numpy as np
import pandas as pd
import time
import navix as nx
from wrappers import LogWrapper, FlattenObservationWrapper, NavixGymnaxWrapper
from utils.utils import L1Norm, L2Norm

class PerceptionRecurrentNetwork(nn.Module):
    """Perception network (follower in SCORER) that manages recurrent state and representation learning"""
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, hidden: jnp.ndarray):
        original_shape = x.shape

        if x.ndim == 1:
            x = x[None, :]

        if x.ndim == 2:
            x = x[:, None, :]

        batch_size, seq_len, obs_dim = x.shape

        if hidden.ndim == 1:
            hidden = hidden[None, :]

        # Ensure hidden matches batch size
        if hidden.shape[0] == 1 and batch_size > 1:
            hidden = jnp.broadcast_to(hidden, (batch_size, self.hidden_dim))
        elif hidden.shape[0] != batch_size:
            hidden = hidden[:batch_size] if hidden.shape[0] > batch_size else jnp.broadcast_to(hidden, (batch_size, self.hidden_dim))

        # Embedding layer
        x = L2Norm()(x)
        x = nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.tanh(x)

        gru_cell = nn.GRUCell(features=self.hidden_dim)

        outputs = []
        carry = hidden
        for t in range(seq_len):
            carry, y = gru_cell(carry, x[:, t, :])
            outputs.append(y)

        representation = jnp.stack(outputs, axis=1)

        if len(original_shape) == 1:
            representation = representation[0, 0]
            carry = carry[0]
        elif len(original_shape) == 2 and original_shape[0] == batch_size:
            representation = representation[:, 0, :]
        carry = L2Norm()(carry)
        representation = L2Norm()(representation)
        return representation, carry

class ControlNetwork(nn.Module):
    """Control network (leader in SCORER) that maps representations to Q-values"""
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Takes representation from perception network
        # x = L1Norm()(x)
        x = nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.tanh(x)
        q_values = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return q_values

@chex.dataclass(frozen=True)
class TimeStep:
    """TimeStep for R2D2 with proper episode ending handling."""
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    truncated: chex.Array
    hidden: chex.Array

class PerceptionTrainState(TrainState):
    target_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

class ControlTrainState(TrainState):
    target_params: flax.core.FrozenDict

def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    env_name = config.get("DEPLOY_ENV") if config.get("DEPLOY_ENV") else config["ENV_NAME"]
    env = NavixGymnaxWrapper(env_name, observation_fn=nx.observations.symbolic_first_person)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env_params = None

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

        # Get observation dimension from actual observation
        obs_dim = init_obs.shape[-1]

        # Initialize trajectory buffer
        rng, _rng_buffer = jax.random.split(rng)
        buffer = fbx.make_prioritised_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["MIN_BUFFER_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=config["SEQUENCE_LENGTH"],
            period=config["PERIOD"]
        )

        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=(0,)),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
            set_priorities=jax.jit(buffer.set_priorities, donate_argnums=(0,)),
        )

        _action = env.action_space(env_params).sample(_rng_buffer)
        _obs = jnp.zeros_like(init_obs[0])
        _reward = jnp.array(0.0)
        _done = jnp.array(False)
        _hidden = jnp.zeros(config["HIDDEN_DIM"])
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done, truncated=jnp.array(False), hidden=_hidden)
        buffer_state = buffer.init(_timestep)

        # Initialize perception network
        perception_network = PerceptionRecurrentNetwork(hidden_dim=config["HIDDEN_DIM"])
        rng, _rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((obs_dim,))
        dummy_hidden = jnp.zeros((config["HIDDEN_DIM"],))
        perception_params = perception_network.init(_rng, dummy_obs, dummy_hidden)

        # Initialize control network
        control_network = ControlNetwork(action_dim=env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        dummy_representation, _ = perception_network.apply(perception_params, dummy_obs, dummy_hidden)
        control_params = control_network.init(_rng, dummy_representation)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        def perception_linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["PERCEPTION_LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        perception_lr = perception_linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["PERCEPTION_LR"]

        # Separate optimizers for perception (follower) and control (leader)
        perception_tx = optax.chain(
            optax.clip_by_global_norm(config.get("PERCEPTION_MAX_GRAD_NORM", 0.5)),
            optax.adam(learning_rate=perception_lr)
        )

        control_tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=lr)
        )

        perception_state = PerceptionTrainState.create(
            apply_fn=perception_network.apply,
            params=perception_params,
            target_params=jax.tree.map(lambda x: jnp.copy(x), perception_params),
            tx=perception_tx,
            timesteps=0,
            n_updates=0,
        )

        control_state = ControlTrainState.create(
            apply_fn=control_network.apply,
            params=control_params,
            target_params=jax.tree.map(lambda x: jnp.copy(x), control_params),
            tx=control_tx,
        )

        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(rng, 2)
            eps = jnp.clip(
                ((config["EPSILON_FINISH"] - config["EPSILON_START"]) / config["EPSILON_ANNEAL_TIME"]) * t + config["EPSILON_START"],
                config["EPSILON_FINISH"],
                config["EPSILON_START"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)
            chosen_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape) < eps,
                jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
                greedy_actions,
            )
            return chosen_actions

        # Initialize hidden states for each environment
        hidden_states = jnp.zeros((config["NUM_ENVS"], config["HIDDEN_DIM"]))

        # Storage for sequences - we need to batch timesteps before adding to buffer
        sequence_buffer = {
            'obs': jnp.zeros((config["NUM_ENVS"], config["BUFFER_ADD_SEQUENCE_LENGTH"], obs_dim), dtype=init_obs.dtype),
            'action': jnp.zeros((config["NUM_ENVS"], config["BUFFER_ADD_SEQUENCE_LENGTH"]), dtype=jnp.int32),
            'reward': jnp.zeros((config["NUM_ENVS"], config["BUFFER_ADD_SEQUENCE_LENGTH"])),
            'done': jnp.zeros((config["NUM_ENVS"], config["BUFFER_ADD_SEQUENCE_LENGTH"]), dtype=bool),
            'truncated': jnp.zeros((config["NUM_ENVS"], config["BUFFER_ADD_SEQUENCE_LENGTH"]), dtype=bool),
            'hidden': jnp.zeros((config["NUM_ENVS"], config["BUFFER_ADD_SEQUENCE_LENGTH"], config["HIDDEN_DIM"]))
        }
        seq_idx = 0

        def _update_step(runner_state, unused):
            perception_state, control_state, buffer_state, env_state, last_obs, hidden_states, sequence_buffer, seq_idx, rng = runner_state

            rng, rng_a, rng_s = jax.random.split(rng, 3)

            representation, new_hidden = perception_network.apply(perception_state.params, last_obs, hidden_states)

            q_vals = control_network.apply(control_state.params, representation)
            action = eps_greedy_exploration(rng_a, q_vals, perception_state.timesteps)

            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(rng_s, env_state, action)

            # Store current step in sequence buffer
            sequence_buffer['obs'] = sequence_buffer['obs'].at[:, seq_idx].set(last_obs)
            sequence_buffer['action'] = sequence_buffer['action'].at[:, seq_idx].set(action)
            sequence_buffer['reward'] = sequence_buffer['reward'].at[:, seq_idx].set(reward)
            sequence_buffer['done'] = sequence_buffer['done'].at[:, seq_idx].set(done)
            sequence_buffer['truncated'] = sequence_buffer['truncated'].at[:, seq_idx].set(info.get('truncation', jnp.zeros_like(done)))
            sequence_buffer['hidden'] = sequence_buffer['hidden'].at[:, seq_idx].set(hidden_states)

            # Reset hidden states where episodes ended
            hidden_states = jnp.where(done[:, None], jnp.zeros_like(new_hidden), new_hidden)

            seq_idx = seq_idx + 1

            # Add sequences to buffer when we have enough timesteps
            def add_to_buffer(buffer_state):
                timesteps_batch = TimeStep(
                    obs=sequence_buffer['obs'],
                    action=sequence_buffer['action'],
                    reward=sequence_buffer['reward'],
                    done=sequence_buffer['done'],
                    truncated=sequence_buffer['truncated'],
                    hidden=sequence_buffer['hidden']
                )
                return buffer.add(buffer_state, timesteps_batch)

            buffer_state = jax.lax.cond(
                seq_idx >= config["BUFFER_ADD_SEQUENCE_LENGTH"],
                add_to_buffer,
                lambda x: x,
                buffer_state
            )

            # Reset sequence buffer and index when full
            sequence_buffer = jax.tree.map(
                lambda x: jax.lax.cond(seq_idx >= config["BUFFER_ADD_SEQUENCE_LENGTH"], lambda: jnp.zeros_like(x), lambda: x),
                sequence_buffer
            )
            seq_idx = jax.lax.cond(seq_idx >= config["BUFFER_ADD_SEQUENCE_LENGTH"], lambda: 0, lambda: seq_idx)

            perception_state = perception_state.replace(timesteps=perception_state.timesteps + config["NUM_ENVS"])

            def _learn_phase(perception_state, control_state, buffer_state, rng):

                # Check if it's time for control network update
                is_control_update_time = (perception_state.timesteps % config["CONTROL_UPDATE_INTERVAL"] == 0)

                # CONTROL (Leader)
                def control_step(carry, _):
                    control_state_inner, rng_inner = carry
                    rng_inner, rng_sample = jax.random.split(rng_inner)

                    # Sample new batch for each follower step
                    sample = buffer.sample(buffer_state, rng_sample)
                    learn_batch = sample.experience
                    probs = sample.probabilities
                    indices = sample.indices
                    batch_size, seq_len = learn_batch.obs.shape[:2]
                    burn_in_length = config["BURN_IN_LENGTH"]
                    learning_length = seq_len - burn_in_length

                    # Get initial hidden states after burn-in
                    burn_in_obs = learn_batch.obs[:, :burn_in_length]
                    initial_hiddens = learn_batch.hidden[:, 0]
                    burn_in_dones = learn_batch.done[:, :burn_in_length]

                    def process_burn_in_with_resets(params, obs_seq, done_seq, initial_hidden):
                        def scan_fn(hidden, inputs):
                            obs, done = inputs
                            # Pass single observation with batch dim
                            _, new_hidden = perception_network.apply(params, obs[None, :], hidden[None, :])
                            new_hidden = new_hidden[0]  # Remove batch dim
                            # Reset hidden state if done
                            hidden = jax.lax.cond(
                                done,
                                lambda: jnp.zeros_like(new_hidden),
                                lambda: new_hidden
                            )
                            return hidden, None

                        final_hidden, _ = jax.lax.scan(scan_fn, initial_hidden, (obs_seq, done_seq))
                        return final_hidden

                    # Burn-in using perception network
                    burn_in_hidden = jax.vmap(process_burn_in_with_resets, in_axes=(None, 0, 0, 0))(
                        perception_state.params, burn_in_obs, burn_in_dones, initial_hiddens
                    )

                    # Get data for learning period
                    learning_obs = learn_batch.obs[:, burn_in_length:]
                    learning_actions = learn_batch.action[:, burn_in_length:]
                    learning_rewards = learn_batch.reward[:, burn_in_length:burn_in_length+learning_length-1]
                    learning_dones = learn_batch.done[:, burn_in_length:burn_in_length+learning_length-1]
                    learning_truncated = learn_batch.truncated[:, burn_in_length:burn_in_length+learning_length-1]

                    def follower_loss_fn(control_params):
                        # Get representations from perception network
                        representations, _ = perception_network.apply(
                            perception_state.params, learning_obs[:, :-1], burn_in_hidden
                        )

                        representations = jax.lax.stop_gradient(representations)

                        # Get Q-values from control network
                        q_vals = control_network.apply(control_params, representations)
                        q_selected = jnp.take_along_axis(
                            q_vals,
                            learning_actions[:, :-1, None],
                            axis=-1
                        ).squeeze(axis=-1)

                        representations_next, _ = perception_network.apply(
                            perception_state.target_params, learning_obs, burn_in_hidden
                        )
                        representations_next = jax.lax.stop_gradient(representations_next)

                        q_next_online = control_network.apply(control_state_inner.params, representations_next)
                        q_next_target = control_network.apply(control_state_inner.target_params, representations_next)

                        next_actions = jnp.argmax(q_next_online[:, 1:], axis=-1)
                        q_next_target_selected = jnp.take_along_axis(
                            q_next_target[:, 1:],
                            next_actions[..., None],
                            axis=-1
                        ).squeeze(axis=-1)

                        discounts = jnp.where(
                            learning_truncated,
                            config["GAMMA"],
                            (1.0 - learning_dones) * config["GAMMA"]
                        )

                        def batch_n_step(rewards, discounts, q_values):
                            return rlax.n_step_bootstrapped_returns(
                                rewards,
                                discounts,
                                q_values,
                                config["N_STEPS"]
                            )

                        targets = jax.vmap(batch_n_step)(
                            learning_rewards,
                            discounts,
                            q_next_target_selected
                        )

                        td_error = targets - q_selected

                        # Importance sampling for PER
                        importance = (1.0 / (probs + 1e-6))
                        beta = jnp.clip(
                            config["PER_BETA_START"] + (config["PER_BETA_END"] - config["PER_BETA_START"]) *
                            (perception_state.timesteps / config["TOTAL_TIMESTEPS"]),
                            config["PER_BETA_START"],
                            config["PER_BETA_END"]
                        )
                        importance = importance ** beta
                        importance = importance / jnp.max(importance)

                        loss = jnp.mean(importance[:, None] * jnp.square(td_error))

                        # Calculate priorities
                        abs_td = jnp.abs(td_error) + config["PER_EPSILON"]
                        priorities = jnp.max(abs_td, axis=1) ** config["PER_ALPHA"]

                        return loss, (priorities, targets, indices)

                    (control_loss, (priorities, targets, indices)), control_grads = jax.value_and_grad(
                        follower_loss_fn, has_aux=True
                    )(control_state_inner.params)

                    control_state_inner = control_state_inner.apply_gradients(grads=control_grads)

                    return (control_state_inner, rng_inner), (control_loss, priorities, targets, indices)

                def do_control_updates():
                    # Perform multiple gradient steps for control network to commit
                    return jax.lax.scan(
                        control_step,
                        (control_state, rng),
                        None,
                        config["CONTROL_CONVERGENCE_STEPS"]
                    )

                def skip_control_updates():
                    # Return unchanged state with dummy metrics
                    dummy_losses = jnp.zeros((config["CONTROL_CONVERGENCE_STEPS"],))
                    dummy_priorities = jnp.zeros((config["CONTROL_CONVERGENCE_STEPS"], config["BUFFER_BATCH_SIZE"]))
                    dummy_targets = jnp.zeros((config["CONTROL_CONVERGENCE_STEPS"], config["BUFFER_BATCH_SIZE"], config["SEQUENCE_LENGTH"] - config["BURN_IN_LENGTH"] - 1))
                    dummy_indices = jnp.zeros((config["CONTROL_CONVERGENCE_STEPS"], config["BUFFER_BATCH_SIZE"]), dtype=jnp.int32)
                    return (control_state, rng), (dummy_losses, dummy_priorities, dummy_targets, dummy_indices)

                (control_state, _), (control_losses, priorities_list, targets_list, indices_list) = jax.lax.cond(
                    is_control_update_time,
                    lambda: do_control_updates(),
                    lambda: skip_control_updates()
                )

                if config["CONTROL_CONVERGENCE_STEPS"] > 1:
                    control_loss = control_losses[-1]
                    priorities = priorities_list[-1]
                    targets = targets_list[-1]
                    indices = indices_list[-1]
                else:
                    control_loss = jnp.squeeze(control_losses)
                    priorities = jnp.squeeze(priorities_list, axis=0)
                    targets = jnp.squeeze(targets_list, axis=0)
                    indices = jnp.squeeze(indices_list, axis=0)

                perception_state = perception_state.replace(n_updates=perception_state.n_updates + 1)

                buffer_state = jax.lax.cond(
                    is_control_update_time,
                    lambda: buffer.set_priorities(buffer_state, indices, priorities),
                    lambda: buffer_state
                )

                # PERCEPTION UPDATE - Sample new batch for perception
                rng, rng_leader = jax.random.split(rng)
                leader_sample = buffer.sample(buffer_state, rng_leader)
                leader_batch = leader_sample.experience
                leader_batch_size, leader_seq_len = leader_batch.obs.shape[:2]
                leader_burn_in_length = config["BURN_IN_LENGTH"]
                leader_learning_length = leader_seq_len - leader_burn_in_length

                # Process leader batch
                leader_burn_in_obs = leader_batch.obs[:, :leader_burn_in_length]
                leader_initial_hiddens = leader_batch.hidden[:, 0]
                leader_burn_in_dones = leader_batch.done[:, :leader_burn_in_length]

                def process_burn_in_with_resets(params, obs_seq, done_seq, initial_hidden):
                    def scan_fn(hidden, inputs):
                        obs, done = inputs
                        _, new_hidden = perception_network.apply(params, obs[None, :], hidden[None, :])
                        new_hidden = new_hidden[0]
                        # Reset hidden state if done
                        hidden = jax.lax.cond(
                            done,
                            lambda: jnp.zeros_like(new_hidden),
                            lambda: new_hidden
                        )
                        return hidden, None

                    final_hidden, _ = jax.lax.scan(scan_fn, initial_hidden, (obs_seq, done_seq))
                    return final_hidden

                # Burn-in using perception network
                leader_burn_in_hidden = jax.vmap(process_burn_in_with_resets, in_axes=(None, 0, 0, 0))(
                    perception_state.params, leader_burn_in_obs, leader_burn_in_dones, leader_initial_hiddens
                )

                # Get leader data for learning period
                leader_learning_obs = leader_batch.obs[:, leader_burn_in_length:]
                leader_learning_actions = leader_batch.action[:, leader_burn_in_length:]
                leader_learning_rewards = leader_batch.reward[:, leader_burn_in_length:leader_burn_in_length+leader_learning_length-1]
                leader_learning_dones = leader_batch.done[:, leader_burn_in_length:leader_burn_in_length+leader_learning_length-1]
                leader_learning_truncated = leader_batch.truncated[:, leader_burn_in_length:leader_burn_in_length+leader_learning_length-1]

                def perception_loss_fn(perception_params):
                    # Get representations using leader's parameters
                    representations, _ = perception_network.apply(
                        perception_params, leader_learning_obs[:, :-1], leader_burn_in_hidden
                    )

                    # Using fixed control network parameters (true Stackelberg)
                    q_vals = control_network.apply(
                        jax.lax.stop_gradient(control_state.params),
                        representations
                    )
                    q_selected = jnp.take_along_axis(
                        q_vals,
                        leader_learning_actions[:, :-1, None],
                        axis=-1
                    ).squeeze(axis=-1)

                    # Compute targets for leader batch
                    representations_next, _ = perception_network.apply(
                        perception_state.target_params, leader_learning_obs, leader_burn_in_hidden
                    )
                    representations_next = jax.lax.stop_gradient(representations_next)

                    q_next_online = control_network.apply(control_state.params, representations_next)
                    q_next_target = control_network.apply(control_state.target_params, representations_next)

                    next_actions = jnp.argmax(q_next_online[:, 1:], axis=-1)
                    q_next_target_selected = jnp.take_along_axis(
                        q_next_target[:, 1:],
                        next_actions[..., None],
                        axis=-1
                    ).squeeze(axis=-1)

                    discounts = jnp.where(
                        leader_learning_truncated,
                        config["GAMMA"],
                        (1.0 - leader_learning_dones) * config["GAMMA"]
                    )

                    def batch_n_step(rewards, discounts, q_values):
                        return rlax.n_step_bootstrapped_returns(
                            rewards,
                            discounts,
                            q_values,
                            config["N_STEPS"]
                        )

                    leader_targets = jax.vmap(batch_n_step)(
                        leader_learning_rewards,
                        discounts,
                        q_next_target_selected
                    )

                    td_errors = leader_targets - q_selected

                    # Calculate loss components
                    msbe = jnp.mean(jnp.square(td_errors))
                    td_mean = jnp.mean(td_errors)
                    be_variance = jnp.mean(jnp.square(td_errors - td_mean))

                    # Combine loss components based on config
                    loss = msbe * (1.0 - config["USE_VARIANCE_LOSS"])

                    if config["USE_VARIANCE_LOSS"]:
                        loss = loss + be_variance


                    return loss, {
                        "be_variance": be_variance,
                        "msbe": msbe
                    }

                is_perception_update_time = (perception_state.timesteps % config["PERCEPTION_UPDATE_INTERVAL"] == 0)

                def update_perception():
                    (perception_loss, perception_metrics), perception_grads = jax.value_and_grad(
                        perception_loss_fn, has_aux=True
                    )(perception_state.params)
                    return perception_state.apply_gradients(grads=perception_grads), perception_metrics

                def skip_perception_update():
                    _, perception_metrics = perception_loss_fn(perception_state.params)
                    return perception_state, perception_metrics

                perception_state, perception_metrics = jax.lax.cond(
                    is_perception_update_time,
                    lambda: update_perception(),
                    lambda: skip_perception_update()
                )

                return perception_state, control_state, buffer_state, control_loss, perception_metrics

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (perception_state.timesteps > config["LEARNING_STARTS"])
                & (perception_state.timesteps % jnp.minimum(config["CONTROL_UPDATE_INTERVAL"], config["PERCEPTION_UPDATE_INTERVAL"]) == 0)
            )

            empty_perception_metrics = {
                "be_variance": jnp.array(0.0),
                "msbe": jnp.array(0.0)
            }

            perception_state, control_state, buffer_state, control_loss, perception_metrics = jax.lax.cond(
                is_learn_time,
                lambda args: _learn_phase(args[0], args[1], args[2], args[3]),
                lambda args: (args[0], args[1], args[2], jnp.array(0.0), empty_perception_metrics),
                (perception_state, control_state, buffer_state, _rng),
            )

            # Update target networks
            def update_targets():
                new_perception_targets = optax.incremental_update(
                    perception_state.params,
                    perception_state.target_params,
                    config["TAU"]
                )
                new_control_targets = optax.incremental_update(
                    control_state.params,
                    control_state.target_params,
                    config["TAU"]
                )
                return (
                    perception_state.replace(target_params=new_perception_targets),
                    control_state.replace(target_params=new_control_targets)
                )

            def skip_target_update():
                return perception_state, control_state

            perception_state, control_state = jax.lax.cond(
                perception_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda: update_targets(),
                lambda: skip_target_update()
            )

            metrics = {
                "timesteps": perception_state.timesteps,
                "updates": perception_state.n_updates,
                "control_loss": control_loss.mean(),
                "perception_loss": perception_metrics.get("msbe", 0.0),
                "be_variance": perception_metrics.get("be_variance", 0.0),
                "returns": info["returned_episode_returns"].mean(),
            }

            runner_state = (perception_state, control_state, buffer_state, env_state, obs, hidden_states, sequence_buffer, seq_idx, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (perception_state, control_state, buffer_state, env_state, init_obs, hidden_states, sequence_buffer, seq_idx, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metrics}

    return train

def run_single_seed(config, seed):
    if config.get("WANDB_MODE", "disabled") == "online":
        wandb.init(
            project=config.get("PROJECT", "SCORER_R2D2"),
            tags=["SCORER_R2D2", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'SCORER_R2D2_{config["ENV_NAME"]}_{seed}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(seed)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    print(f"Starting training for seed {seed}...")
    out = train_jit(rng)

    if config.get("WANDB_MODE", "disabled") == "online":
        returns = np.array(out["metrics"]["returns"])
        timesteps = np.array(out["metrics"]["timesteps"])
        losses = np.array(out["metrics"]["control_loss"])

        for i, (ts, ret, loss) in enumerate(zip(timesteps, returns, losses)):
            if i % 100 == 0:
                wandb.log({"returns": ret, "timesteps": ts, "loss": loss})
        wandb.finish()

    returns = np.array(out["metrics"]["returns"])
    timesteps = np.array(out["metrics"]["timesteps"])
    os.makedirs("results", exist_ok=True)

    loss_components = []
    if config["USE_MSBE"]:
        loss_components.append("MSBE")
    if config["USE_VARIANCE_LOSS"]:
        loss_components.append("BEVar")
    loss_str = '+'.join(loss_components) if loss_components else "MSBE"

    filename_prefix = f"SCORER_R2D2_{config['ENV_NAME']}_{loss_str}_{seed}"
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
    os.makedirs("results/SCORER_R2D2", exist_ok=True)

    all_returns = outs["metrics"]["returns"]
    all_timesteps = outs["metrics"]["timesteps"][0]
    all_losses = outs["metrics"]["control_loss"]

    loss_components = []
    if config["USE_MSBE"]:
        loss_components.append("MSBE")
    if config["USE_VARIANCE_LOSS"]:
        loss_components.append("BEVar")

    loss_str = '+'.join(loss_components) if loss_components else "MSBE"

    filename_prefix = f"SCORER_R2D2/{config['ENV_NAME']}_{loss_str}_{config['SEED']}"

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

args = parse_args()

config = {
    "NUM_SEEDS": args.num_seeds,

    # R2D2 specific parameters
    "HIDDEN_DIM": 128,
    "SEQUENCE_LENGTH": 100,
    "BUFFER_ADD_SEQUENCE_LENGTH": 32,
    "BURN_IN_LENGTH": 50,
    "PERIOD": 4,
    "N_STEPS": 10,

    # Buffer parameters
    "NUM_ENVS": 10,
    "BUFFER_SIZE": 100_000,
    "MIN_BUFFER_SIZE": 5000,
    "BUFFER_BATCH_SIZE": 32,

    # Training parameters
    "TOTAL_TIMESTEPS": args.total_timesteps,
    "EPSILON_START": 1.0,
    "EPSILON_FINISH": 0.01,
    "EPSILON_ANNEAL_TIME": 500_000,
    "TARGET_UPDATE_INTERVAL": 5_000,
    "LR": 1e-4,
    "LEARNING_STARTS": 25_000,
    "CONTROL_UPDATE_INTERVAL": 16,
    "LR_LINEAR_DECAY": True,
    "GAMMA": 0.99,
    "TAU": 1.0,

    # SCORER specific parameters
    "PERCEPTION_LR": 3e-4,  # Perception (follower) has higher learning rate
    "PERCEPTION_UPDATE_INTERVAL": 16,
    "PERCEPTION_MAX_GRAD_NORM": 0.5,
    "CONTROL_CONVERGENCE_STEPS": args.control_convergence_steps, # following TTSA we stick to 1 step
    "USE_MSBE": not (args.use_be_variance),
    "USE_VARIANCE_LOSS": args.use_be_variance,

    # PER parameters
    "PER_ALPHA": 0.6,
    "PER_BETA_START": 0.4,
    "PER_BETA_END": 1.0,
    "PER_EPSILON": 1e-6,

    # Environment and logging
    "ENV_NAME": args.env,
    "SEED": args.seed,
    "WANDB_MODE": "disabled",
    "PROJECT": "SCORER_R2D2",
    "DEBUG": False,
}

if __name__ == "__main__":
    print("Available devices:", jax.devices())

    if config["NUM_SEEDS"] == 1:
        print("Running single seed training...")
        out = run_single_seed(config, config["SEED"])
    else:
        print(f"Running training with {config['NUM_SEEDS']} seeds in parallel...")
        outs = run_multiple_seeds(config)
        print("Multi-seed training complete!")