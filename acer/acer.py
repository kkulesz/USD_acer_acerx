from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Iterable

import gym
import numpy as np
import torch
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from .acer_utils.replay_buffer import ACERReplayBuffer

from torch.utils.tensorboard import SummaryWriter


def get_parameters_by_name(model: th.nn.Module, included_names: Iterable[str]) -> List[th.Tensor]:
    """
    Extract parameters from the state dict of ``model``
    if the name contains one of the strings in ``included_names``.
    :param model: the model where the parameters come from.
    :param included_names: substrings of names to include.
    :return: List of parameters values (Pytorch tensors)
        that matches the queried names.
    """
    return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]


class ACER(OffPolicyAlgorithm):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 200,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ACERReplayBuffer] = ACERReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            lam: float = 0.1,
            b: float = 3,
            c: int = 10,
            c0: float = 0.3,
            trajectory_lens: int = 10
    ):

        super(ACER, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )
        self._lam = lam
        self._b = b
        self._c = c
        self._c0 = c0
        self._trajectory_lens = trajectory_lens
        print(self.train_freq)
        if _init_setup_model:
            self._setup_model()

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.policy.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data, lens = self.replay_buffer.sample(batch_size, self._trajectory_lens)
            obs_next = replay_data.next_observations[np.arange(batch_size), lens - 1]
            dones = replay_data.dones[np.arange(batch_size), lens - 1].bool()

            dones = 1 - ((th.unsqueeze(th.arange(1, self._trajectory_lens + 1, device=lens.device), 0) == th.unsqueeze(
                lens, 1)) & th.unsqueeze(dones, 1)).float()

            values_next = self.policy.predict_values(obs_next) * dones
            flat_obs = replay_data.observations.reshape(-1, replay_data.observations.shape[-1])
            flat_actions = replay_data.actions.reshape(-1, replay_data.actions.shape[-1])

            values, log_prob, _ = self.policy.evaluate_actions(flat_obs, flat_actions)
            values = values.reshape(replay_data.rewards.shape)
            log_prob = log_prob.reshape(replay_data.rewards.shape)
            mask = self._sequence_mask(lens, maxlen=self._trajectory_lens, dtype=th.float32)

            policies_masked = log_prob * mask + (~mask) * th.ones_like(log_prob)
            old_policies_masked = replay_data.action_probs * mask + (~mask) * th.ones_like(replay_data.action_probs)

            policies_ratio = th.exp(policies_masked - old_policies_masked)
            policies_ratio = th.clamp(policies_ratio, 0.001, 10000)
            policies_ratio_prod = th.cumprod(policies_ratio, dim=-1)

            truncated_density = th.tanh(policies_ratio_prod / self._b) * self._b
            gamma_coeffs_masked = th.unsqueeze(
                th.pow(self.gamma, th.arange(1, self._trajectory_lens + 1, device=mask.device)), dim=0) * mask
            td_parts = replay_data.rewards + self.gamma * values_next - values
            td_rewards = td_parts * gamma_coeffs_masked
            e = th.sum(th.mul(td_rewards, truncated_density), dim=1).detach()
            e = th.unsqueeze(e, dim=1)

            actor_loss = self._actor_loss(log_prob[:, 0], e)
            critic_loss = self._critic_loss(values[:, 0], e)

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            loss = actor_loss + critic_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # # Clip grad norm
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def _actor_loss(self, log_probs: th.Tensor, d: th.Tensor) -> th.Tensor:
        total_loss = th.mean(-log_probs.unsqueeze(1) * d)
        return total_loss

    def _critic_loss(self, values: th.Tensor, d: th.Tensor) -> th.Tensor:

        loss = th.mean(-values.unsqueeze(1) * d)
        return loss

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "ACER",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            self.gradient_steps = min([round(self._c0 * self.num_timesteps), self._c])

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ACERReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # action_prob = self.policy.get_distribution(self._last_obs).log_prob(actions)
            with th.no_grad():
                _, log_prob, _ = self.policy.evaluate_actions(th.tensor(self._last_obs).to(self.device),
                                                              th.tensor(buffer_actions).to(self.device))
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, log_prob.cpu(), rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
            self,
            replay_buffer: ACERReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            action_probs: np.ndarray,  # acer needed
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            action_probs,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
