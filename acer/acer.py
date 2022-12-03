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
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from .policies import ActorCriticPolicy
from .acer_utils.replay_buffer import ACERReplayBuffer


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
    # defaultowe parametry ustawić
    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 100,
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
            c0: float = 0.3
    ):

        super(ACER, self).__init__(
            policy,
            env,
            ActorCriticPolicy,
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
        print(self.train_freq)
        if _init_setup_model:
            self._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.policy.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            values_next = self.policy.predict_values(replay_data.next_observations) * (1.0 - replay_data.dones)
            values, log_prob, _ = self.policy.evaluate_actions(replay_data.observations, replay_data.actions)
            policies = th.exp(log_prob)

            policies_ratio = policies / th.exp(replay_data.action_probs)
            # policies_ratio_product = th.cumprod(policies_ratio, dim=1)

            b_tensor = th.ones_like(policies_ratio) * self._b
            truncated_densities = th.minimum(policies_ratio, b_tensor)

            gamma_coeffs = th.ones_like(truncated_densities) * self.gamma
            # gamma_coeffs = th.cumprod(gamma_coeffs, dim=1)


            d_coeffs = gamma_coeffs * (replay_data.rewards + self.gamma * values_next - values) * truncated_densities

            d = th.sum(d_coeffs, dim=1)
            actor_loss = self._actor_loss(replay_data.observations, replay_data.action_probs, d)
            critic_loss = self._critic_loss(values, d)
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

    def _actor_loss(self, observations: th.Tensor, log_probs: th.Tensor, d: th.Tensor) -> th.Tensor:
        # mean = self.policy.get_distribution(observations).mean_actions
        # print(log_probs.shape, d.shape)
        # out_of_bound_error = th.zeros_like(mean)
        # out_of_bound_error[mean > self.action_space.high] = mean[mean > self.action_space.high] - self.action_space.high
        # out_of_bound_error[mean < self.action_space.low] = self.action_space.low - mean[mean < self.action_space.low]
        # bounds_penalty = th.sum(
        #                     th.square(
        #                         th.maximum(th.tensor(0.), out_of_bound_error)
        #                     ) * 0.001, dim=1
        #                 ) # scale actions so this is not necessary

        # total_loss = th.mean(-log_probs*d + bounds_penalty)
        total_loss = th.mean(-log_probs * d)
        return total_loss


    def _critic_loss(self, values: th.Tensor, d: th.Tensor) -> th.Tensor:
        loss = th.mean(values * d)
        return loss

    # old_policies, , first_obs, first_actions (pierwsze obs, action z pierwszego batcha per env), lengths (długości powtarzantch fragmentów),?
    # @tf.function(experimental_relax_shapes=True)
    # def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
    #                                  rewards, first_obs, first_actions, dones, lengths, *args):
    #     """Backward pass with single batch of experience.
    #
    #     Every experience replay requires sequence of experiences with random length, thus we have to use
    #     ragged tensors here.
    #
    #     See Equation (8) and Equation (9) in the paper (1).
    #     """
    #
    #     # raczej nie używamy
    #     obs = self._process_observations(obs)
    #     obs_next = self._process_observations(obs_next)
    #     rewards = self._process_rewards(rewards)
    #
    #     batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
    #     values = tf.squeeze(self._critic.value(obs))
    #
    #     # values_next = 0 jeśli done
    #     values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
    #     policies, log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
    #     policies, log_policies = tf.squeeze(policies), tf.squeeze(log_policies)
    #     indices = tf.expand_dims(batches_indices, axis=2)
    #
    #     # flat tensor
    #     policies_ratio = tf.math.divide(policies, old_policies)
    #     # ragged tensor divided into batches
    #     policies_ratio_batches = tf.squeeze(tf.gather(policies_ratio, indices), axis=2)
    #
    #     # cumprod and cumsum do not work on ragged tensors, we transform them into tensors
    #     # padded with 0 and then apply boolean mask to retrieve original ragged tensor
    #     batch_mask = tf.sequence_mask(policies_ratio_batches.row_lengths())
    #     policies_ratio_product = tf.math.cumprod(policies_ratio_batches.to_tensor(), axis=1)
    #
    #     # min
    #     # {∏k
    #     # j=0
    #     # π (ai+j ;si+j ,θ )
    #     # π (ai+j ;si+j ,θi+j ) , b
    #     truncated_densities = tf.ragged.boolean_mask(
    #         tf.minimum(policies_ratio_product, self._b),
    #         batch_mask
    #     )
    #     gamma_coeffs_batches = tf.ones_like(truncated_densities).to_tensor() * self._gamma
    #     gamma_coeffs = tf.ragged.boolean_mask(
    #         tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
    #         batch_mask
    #     ).flat_values
    #
    #     # flat tensors di+k = ri+k + γ  ̄V (si+k+1; υ) −  ̄V (si+k; υ)
    #     d_coeffs = gamma_coeffs * (rewards + self._gamma * values_next - values) * truncated_densities.flat_values
    #     # ragged
    #     d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
    #     # final summation over original batches
    #     d = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))
    #
    #     self._backward_pass(first_obs, first_actions, d)
    #
    #     _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
    #     new_log_policies = tf.squeeze(new_log_policies)
    #     approx_kl = tf.reduce_mean(policies - new_log_policies)
    #     with tf.name_scope('actor'):
    #         tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)

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
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

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
        action_probs: np.ndarray, # acer needed
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
