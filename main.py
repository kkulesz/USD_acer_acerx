import torch
import gym
import time
# from acer.acer_classic import ACER
from acer.other_group.acer import ACER
from acer.acer import ACER
from acer.other_group.acerax import ACERAX
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    print(torch.__version__)
    print(gym.__version__)

    train_model(model_path='acerax_ant', algorithm=ACER, policy="MlpPolicy", use_wandb=False)
    visualize_model(saved_model_path='acerax_ant', algorithm=ACER)




def train_model(env_name="Ant-v2", model_path="a2c_ant", algorithm=A2C, policy="MlpPolicy", use_wandb=False):
    env = gym.make(env_name)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    if use_wandb:
        wandb_run = wandb.init(
            project="usd_acer",
            entity="bolber",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
        )
        model = algorithm(policy, env, verbose=1, tensorboard_log=f"logs/{model_path}/{wandb_run.id}")
        model.learn(total_timesteps=200_000, callback=WandbCallback(gradient_save_freq=10))
        wandb_run.finish()
    else:
        model = algorithm(policy, env, verbose=1)
        model.learn(total_timesteps=int(2e5), tb_log_name=model_path + timestamp)

    model.save(model_path)


def visualize_model(env_name="Ant-v2", saved_model_path="a2c_ant", algorithm=A2C):
    env = gym.make(env_name)
    model = algorithm.load(saved_model_path, env=env)
    obs = env.reset()
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()


