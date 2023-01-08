import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from copy import deepcopy
from acer.acer_utils.acerax_replay_buffer_trajectories import ACERAXReplayBuffer
from .acaerax_policy import ACERAXPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    get_linear_fn,
    should_collect_more_steps,
    update_learning_rate,
    get_schedule_fn,
)

from stable_baselines3.common.vec_env import VecEnv

ACERAXSelf = TypeVar("ACERAXSelf", bound="ACERAX")


class ACERAX(OffPolicyAlgorithm):
    """
    The ACER (Actor-Critic with Experience Replay) model class, https://arxiv.org/abs/1611.01224

    !TODO: code copied from DQM. The params prepended with `--` are not yet tested and may not be needed

    --:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    --:param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    --:param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    --:param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    --:param gamma: the discount factor
    --:param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    --:param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    --:param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    --:param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    --:param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    --:param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    --:param exploration_initial_eps: initial value of random action probability
    --:param exploration_final_eps: final value of random action probability
    --:param max_grad_norm: The maximum value for the gradient clipping
    --:param tensorboard_log: the log location for tensorboard (if None, no logging)
    --:param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically (Only available when passing string for the environment).
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    --:param policy_kwargs: additional arguments to be passed to the policy on creation
    --:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    --:param seed: Seed for the pseudo random generators
    --:param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    --:param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ACERAXPolicy,
    }

    policy: ACERAXPolicy
    # actor: Actor
    critic: nn.Module
    replay_buffer: ACERAXReplayBuffer

    def __init__(
            self,
            policy: Union[str, Type[ACERAXPolicy]],
            env: Union[GymEnv, str],
            lr_actor: Union[float, Schedule] = 1e-4,
            lr_std: Union[float, Schedule] = 1e-5,
            lr_critic: Union[float, Schedule] = 1e-5,
            buffer_num_trajectories: int = 2,
            buffer_trajectory_size: int = 1000,  # epizode len
            learning_starts: int = 1000,
            batch_size: int = 32,
            buffer_sample_trajectory_size: int = 8,
            alpha: float = 0.3,  # SUM component
            tau: float = 3.00,  # min{policy_frac, b}; b = tau
            alpha_loss: float = 1.,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 4,
            gradient_steps: int = 1,
            replay_buffer_class: Optional[
                Type[ACERAXReplayBuffer]
            ] = ACERAXReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 1.0,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        print(policy_kwargs)
        policy_kwargs = {"log_std_init": 0.6}
        super().__init__(
            policy=policy,
            env=env,
            # (lr_actor, lr_critic),
            learning_rate=lr_actor,
            buffer_size=100_000,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=None,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box,),
            support_multi_env=True,
        )

        self.replay_buffer = replay_buffer_class(
            buffer_num_trajectories=buffer_num_trajectories,
            trajectory_size=buffer_trajectory_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=False,
        )
        self.buffer_sample_trajectory_size = buffer_sample_trajectory_size
        self.alpha_loss = alpha_loss

        self.lr_actor = lr_actor
        self.lr_std = lr_std
        self.lr_critic = lr_critic
        self.alpha = alpha
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""

        self.actor_lr_schedule = get_schedule_fn(self.lr_actor)
        self.std_lr_schedule = get_schedule_fn(self.lr_std)
        self.critic_lr_schedule = get_schedule_fn(self.lr_critic)
        self.lr_schedule = (self.actor_lr_schedule, self.std_lr_schedule, self.critic_lr_schedule)


    def _create_aliases(self) -> None:
        pass

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate()

        losses = []
        actor_losses, critic_losses, std_losses = [], [], []
        sums = []
        action_mean_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size,
                self.buffer_sample_trajectory_size,
                env=self._vec_normalize_env,
            )

            rolling_pi_coef = th.full(
                (self.batch_size, 1), 1.0, dtype=th.float32, device=self.device
            )
            SUM = th.full(
                (self.batch_size, 1), 0.0, dtype=th.float32, device=self.device
            )

            # The second dimension of the replay data tensors is the in-trajectory index.
            for k in range(replay_data.actions.shape[1]):
                actions = replay_data.actions[:, k]
                log_probs = replay_data.action_probs[:, k]
                observations = replay_data.observations[:, k]
                next_observations = replay_data.next_observations[:, k]
                dones = replay_data.dones[:, k]
                rewards = replay_data.rewards[:, k]

                current_dist = self.policy.get_distribution(observations)
                current_log_probs = current_dist.log_prob(actions)
                current_values = self.policy.predict_values(observations)

                log_probs = log_probs.unsqueeze(-1)
                current_log_probs = current_log_probs.unsqueeze(-1)

                if k == 0:
                    k0_current_dist = current_dist
                    k0_current_log_probs = current_log_probs
                    k0_current_values = current_values

                with th.no_grad():
                    next_values = self.policy.predict_values(next_observations)
                    # Avoid potential broadcast issue
                    next_values = next_values.reshape(-1, 1)

                    target_values = rewards + (1 - dones) * self.gamma * next_values
                    advantage = target_values - current_values

                    exp = th.exp(current_log_probs - log_probs)
                    rolling_pi_coef *= exp
                    rolling_pi_coef = th.clamp(
                        rolling_pi_coef,
                        th.full(
                            size=rolling_pi_coef.shape,
                            fill_value=0.0001,
                            device=self.device,
                        ),
                        th.full(
                            size=rolling_pi_coef.shape,
                            fill_value=10000,
                            device=self.device,
                        ),
                    )
                    clamped_pi_coef = th.minimum(
                        rolling_pi_coef,
                        th.full(
                            size=rolling_pi_coef.shape,
                            fill_value=self.tau,
                            device=self.device,
                        ),
                    )
                    SUM += (self.alpha ** k) * advantage * clamped_pi_coef

            actor_loss = k0_current_log_probs * SUM
            actor_loss = actor_loss.mean()
            critic_loss = k0_current_values * SUM
            critic_loss = critic_loss.mean()
            dispersion_loss = self._dispersion_loss(replay_data.actions, replay_data.action_means)

            self.policy.actor.optimizer_std.zero_grad()
            dispersion_loss.backward()
            self.policy.actor.optimizer_std.step()

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.actor.optimizer.step()



            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            std_losses.append(dispersion_loss.item())
            sums.append(SUM.mean().item())

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/std_loss", np.mean(std_losses))

    def _dispersion_loss(self, actions: np.array, m: np.array) -> th.Tensor:
        mean, log_std, _ = self.policy.actor.get_action_dist_params(th.tensor(self._last_obs).to(self.device))
        std = th.exp(log_std)
        mean = mean.detach()
        alpha = self.alpha_loss
        no_alpha = th.sum(0.5 * th.square((m - mean) * th.reciprocal(std)))
        with_alpha = th.sum(0.5 * th.square((actions - mean) * th.reciprocal(std)))
        total_loss = no_alpha + alpha * with_alpha + (1 + alpha) * th.sum(log_std)
        # reg_loss?
        return total_loss

    def _update_learning_rate(self) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).
        """
        self.logger.record(
            "train/lr_actor", self.actor_lr_schedule(self._current_progress_remaining)
        )
        self.logger.record(
            "train/lr_critic", self.critic_lr_schedule(self._current_progress_remaining)
        )

        if self.policy_class == ACERAXPolicy:
            update_learning_rate(
                self.policy.actor.optimizer,
                self.actor_lr_schedule(self._current_progress_remaining),
            )
            update_learning_rate(
                self.policy.critic.optimizer,
                self.critic_lr_schedule(self._current_progress_remaining),
            )
        else:
            update_learning_rate(
                self.policy.optimizer,
                self.actor_lr_schedule(self._current_progress_remaining),
            )

    def learn(
            self: ACERAXSelf,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "ACERAX",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> ACERAXSelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ACERAXReplayBuffer,
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

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            with th.no_grad():
                _, log_prob, _ = self.policy.evaluate_actions(th.tensor(self._last_obs).to(self.device),
                                                              th.tensor(buffer_actions).to(
                                                                  self.device))  # buffer_actions?
                means, _, _ = self.policy.actor.get_action_dist_params(th.tensor(self._last_obs).to(self.device))
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
            self._store_transition(replay_buffer, buffer_actions, log_prob.cpu(), means.cpu(), new_obs, rewards, dones,
                                   infos)

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
            replay_buffer: ACERAXReplayBuffer,
            buffer_action: np.ndarray,
            action_probs: np.ndarray,  # acer needed
            action_means: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
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
            action_means,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
