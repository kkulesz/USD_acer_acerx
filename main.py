import torch
import gym
from acer.acer_classic import ACER
# from acer.acer_fromppo import ACER
from acer.acerax import ACERAX
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import _policy_registry

def main():
    print(torch.__version__)
    print(gym.__version__)

    train_model(model_path='acerax_ant', algorithm=ACER, policy="MlpPolicy")
    visualize_model(saved_model_path='acerax_ant', algorithm=ACER)




def train_model(env_name="Ant-v2", model_path="a2c_ant", algorithm=A2C, policy="MlpPolicy"):
    env = gym.make(env_name)
    model = algorithm(policy, env, verbose=1)
    model.learn(total_timesteps=int(2e5))

    model.save(model_path)


def visualize_model(env_name="Ant-v2", saved_model_path="a2c_ant", algorithm=A2C):
    env = gym.make(env_name)
    model = algorithm.load(saved_model_path, env=env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()


if __name__ == "__main__":
    main()


