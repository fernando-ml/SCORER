# mod of official Navix example: https://github.com/epignatelli/navix/blob/main/examples/purejaxrl/wrappers.py
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
import navix as nx


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        new_info = {
            "returned_episode_returns": state.returned_episode_returns,
            "returned_episode_lengths": state.returned_episode_lengths,
            "timestep": state.timestep,
            "returned_episode": done,
        }
        return obs, state, reward, done, new_info

class NavixGymnaxWrapper:
    def __init__(self, env_name, **kwargs):
        self._env = nx.make(env_name, **kwargs)

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        # Keep the original shape for CNNs - don't flatten
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=self._env.observation_space.shape,  # Don't flatten!
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )

class DualObservationNavixWrapper:
    """Navix wrapper that provides both full and partial observations from the same state."""

    def __init__(self, env_name, partial_obs_fn=None, full_obs_fn=None, **kwargs):
        # Create single environment instance without specific observation function
        self._env = nx.make(env_name, **kwargs)
        self.partial_obs_fn = partial_obs_fn or nx.observations.symbolic_first_person
        self.full_obs_fn = full_obs_fn or nx.observations.symbolic

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        # Apply both observation functions to the actual state, not timestep
        partial_obs = self.partial_obs_fn(timestep.state)
        full_obs = self.full_obs_fn(timestep.state)
        return (partial_obs, full_obs), timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        # Apply both observation functions to the actual state, not timestep
        partial_obs = self.partial_obs_fn(timestep.state)
        full_obs = self.full_obs_fn(timestep.state)
        return (partial_obs, full_obs), timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        # Return space for partial observation (the one the agent uses)
        dummy_timestep = self._env.reset(jax.random.PRNGKey(0))
        partial_obs = self.partial_obs_fn(dummy_timestep.state)
        return spaces.Box(
            low=0,
            high=255,
            shape=partial_obs.shape,
            dtype=partial_obs.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )

class DualFlattenObservationWrapper(GymnaxWrapper):
    """Flatten dual observations (partial, full) separately."""

    def __init__(self, env):
        super().__init__(env)

    def observation_space(self, params):
        # Return space for the partial observation (what the agent sees)
        return self._env.observation_space(params)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        (partial_obs, full_obs), state = self._env.reset(key, params)
        partial_obs = jnp.reshape(partial_obs, (-1,))
        full_obs = jnp.reshape(full_obs, (-1,))
        return (partial_obs, full_obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        (partial_obs, full_obs), state, reward, done, info = self._env.step(key, state, action, params)
        partial_obs = jnp.reshape(partial_obs, (-1,))
        full_obs = jnp.reshape(full_obs, (-1,))
        return (partial_obs, full_obs), state, reward, done, info

class DualLogWrapper(GymnaxWrapper):
    """Log wrapper that handles dual observations."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        (partial_obs, full_obs), env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return (partial_obs, full_obs), state

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        (partial_obs, full_obs), env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        new_info = {
            "returned_episode_returns": state.returned_episode_returns,
            "returned_episode_lengths": state.returned_episode_lengths,
            "timestep": state.timestep,
            "returned_episode": done,
        }
        return (partial_obs, full_obs), state, reward, done, new_info
