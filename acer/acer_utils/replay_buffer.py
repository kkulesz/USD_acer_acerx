import warnings
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple, Tuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class ACERReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    action_probs: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class ACERReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                              dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.action_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.action_probs.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            action_prob: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.action_probs[self.pos] = np.array(action_prob).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, n_batches: int, trajectory_len: int) -> Tuple[ACERReplayBufferSamples, np.ndarray]:
        if self.size() == 0:
            # empty buffer
            return ACERReplayBufferSamples(th.zeros((n_batches, 0)),
                                           th.zeros((n_batches, 0)),
                                           th.zeros((n_batches, 0)),
                                           th.zeros((n_batches, 0)),
                                           th.zeros((n_batches, 0)),
                                           th.zeros((n_batches, 0))), np.repeat(-1, n_batches)

        sample_indices = np.random.randint(low=0, high=self.size(), size=n_batches)

        lens = np.repeat(trajectory_len, n_batches)
        pointer_ind = np.logical_and(sample_indices < self.pos, self.pos < sample_indices + lens)
        lens[pointer_ind] = self.pos - sample_indices[pointer_ind]

        if self.size() < self.buffer_size:
            current_size_ind = np.logical_and(sample_indices + lens > self.size(),
                                              self.size() < self.buffer_size)
            lens[current_size_ind] = self.size() - sample_indices[current_size_ind]

        pointer_ovf_ind = np.logical_and(sample_indices + lens > self.buffer_size,
                                         self.pos < sample_indices + lens - self.buffer_size)
        lens[pointer_ovf_ind] = (self.pos + self.buffer_size) - sample_indices[pointer_ovf_ind]

        selection = (np.repeat(np.expand_dims(sample_indices, 1), trajectory_len, axis=1) + np.arange(trajectory_len))
        selection = selection % self.buffer_size

        batch, lens = self._get_samples(selection, lens)

        return batch, lens

    def _get_samples(self, batch_inds: np.ndarray, lens: np.ndarray, env: Optional[VecNormalize] = None) -> Tuple[
        ACERReplayBufferSamples, th.Tensor]:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=batch_inds.shape)
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices]))
        data = [
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.action_probs[batch_inds, env_indices],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones,
            self._normalize_reward(self.rewards[batch_inds, env_indices], env),
        ]

        ends_mask = np.cumsum(dones[:, :-1], axis=1) == 0

        mask = np.concatenate([np.ones((batch_inds.shape[0], 1)), ends_mask], axis=1)
        for i in range(len(data)):
            m = mask
            while len(m.shape) < len(data[i].shape):
                m = np.expand_dims(m, axis=-1)
            data[i] *= m
        lens = np.minimum(lens, ends_mask.sum(axis=1) + 1)

        return ACERReplayBufferSamples(*tuple(map(self.to_torch, data))), self.to_torch(lens)


