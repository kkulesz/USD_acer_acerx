import warnings
from typing import Any, Dict, List, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.utils import get_device

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class ACERAXReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    action_probs: th.Tensor
    action_means: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class ACERAXReplayBuffer:
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_num_trajectories: int,
        trajectory_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        self.num_trajectories = buffer_num_trajectories
        self.trajectory_size = trajectory_size

        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.trajectory_idx = 0
        self.trajectory_pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        buffer_part_shape = (self.num_trajectories, self.trajectory_size, self.n_envs)

        self.observations = np.zeros(
            buffer_part_shape + self.obs_shape,
            dtype=observation_space.dtype,
        )

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                buffer_part_shape + self.obs_shape,
                dtype=observation_space.dtype,
            )

        self.actions = np.zeros(
            buffer_part_shape + (self.action_dim,), dtype=action_space.dtype
        )

        self.action_means = np.zeros(
            buffer_part_shape + (self.action_dim,), dtype=action_space.dtype
        )

        self.log_probs = np.zeros(buffer_part_shape, dtype=np.float32)
        self.rewards = np.zeros(buffer_part_shape, dtype=np.float32)
        self.dones = np.zeros(buffer_part_shape, dtype=np.float32)

        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.log_probs.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.action_means.nbytes
            )

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
        log_prob: np.ndarray,
        means: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            raise NotImplementedError()

        # Same, for actions
        action = action.reshape((self.n_envs, self.action_dim))
        means = means.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.trajectory_idx, self.trajectory_pos] = np.array(
            obs
        ).copy()

        if self.optimize_memory_usage:
            raise NotImplementedError()
        else:
            self.next_observations[self.trajectory_idx, self.trajectory_pos] = np.array(
                next_obs
            ).copy()

        self.actions[self.trajectory_idx, self.trajectory_pos] = np.array(action).copy()
        self.action_means[self.trajectory_idx, self.trajectory_pos] = np.array(means).copy()
        self.log_probs[self.trajectory_idx, self.trajectory_pos] = np.array(
            log_prob
        ).copy()
        self.rewards[self.trajectory_idx, self.trajectory_pos] = np.array(reward).copy()
        self.dones[self.trajectory_idx, self.trajectory_pos] = np.array(done).copy()

        self.trajectory_pos += 1
        if self.trajectory_pos == self.trajectory_size:
            self.trajectory_idx += 1
            self.trajectory_pos = 0
            if self.trajectory_idx == self.num_trajectories:
                self.full = True
                self.trajectory_idx = 0

    def sample(
        self, batch_size: int, trajectory_size: int, env: Optional[VecNormalize] = None
    ) -> ACERAXReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of trajectories to sample
        :param trajectory_size: Number of elements in each trajectory to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        assert trajectory_size <= self.trajectory_size

        if self.optimize_memory_usage:
            raise NotImplementedError()

        upper_bound = self.num_trajectories - 1 if self.full else self.trajectory_idx
        is_curr_trajectory_big_enough = self.trajectory_pos > batch_size
        if is_curr_trajectory_big_enough:
            upper_bound += 1
        trajectory_inds = np.random.randint(0, upper_bound, size=batch_size)

        # We later clamp the trajectory positions to <0, self.trajectory_size-trajectory_size>.
        # By random sampling from this range and later clamping we prioritize starting and ending trajectories.
        trajectory_positions = np.random.randint(
            -trajectory_size + 1, self.trajectory_size, size=batch_size
        )
        unfinished_trajectory_inds = trajectory_inds == self.trajectory_idx

        trajectory_positions[unfinished_trajectory_inds] = np.minimum(
            trajectory_positions[unfinished_trajectory_inds],
            self.trajectory_pos - trajectory_size - 1,
        )
        trajectory_positions[~unfinished_trajectory_inds] = np.minimum(
            trajectory_positions[~unfinished_trajectory_inds],
            self.trajectory_size - trajectory_size - 1,
        )
        trajectory_positions = np.maximum(trajectory_positions, 0)

        # create the indices of the whole trajectories
        trajectory_positions = np.linspace(
            trajectory_positions,
            trajectory_positions + trajectory_size - 1,
            trajectory_size,
            dtype=np.int,
        ).T

        # Sample randomly the env idx
        env_indices = np.random.randint(
            0, high=self.n_envs, size=(len(trajectory_inds),)
        )

        # allow broadcasting of index arrays
        trajectory_inds = trajectory_inds[:, np.newaxis]
        env_indices = env_indices[:, np.newaxis]

        if self.optimize_memory_usage:
            raise NotImplementedError()
        else:
            next_obs = self._normalize_obs(
                self.next_observations[
                    trajectory_inds,
                    trajectory_positions,
                    env_indices,
                    :,
                ],
                env,
            )

        data = (
            self._normalize_obs(
                self.observations[
                    trajectory_inds, trajectory_positions, env_indices, :
                ],
                env,
            ),
            self.actions[trajectory_inds, trajectory_positions, env_indices, :],
            self.log_probs[trajectory_inds, trajectory_positions, env_indices],
            self.action_means[trajectory_inds, trajectory_positions, env_indices, :],
            next_obs,
            (self.dones[trajectory_inds, trajectory_positions, env_indices])[
                ..., np.newaxis
            ],
            self._normalize_reward(
                self.rewards[trajectory_inds, trajectory_positions, env_indices][
                    ..., np.newaxis
                ],
                env,
            ),
        )
        return ACERAXReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(
        reward: np.ndarray, env: Optional[VecNormalize] = None
    ) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)
