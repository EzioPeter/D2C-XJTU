import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))
from gym.spaces import Discrete, Box
from d2c.networks_and_utils_for_agent.dqn_discrete import make_config
from d2c.models import make_agent
from d2c.trainers import Trainer
from d2c.evaluators import bm_eval
from d2c.envs.external import gym_env
import numpy as np
import wandb

def patch_config_for_discrete_env(cfg, env):
    act_space = env.action_space
    obs_space = env.observation_space

    if isinstance(obs_space, Box):
        cfg.model_config.state_dim = obs_space.shape[0]
    else:
        cfg.model_config.state_dim = 1

    if isinstance(act_space, Discrete):
        cfg.model_config.action_dim = act_space.n
        cfg.model_config.action_type = "discrete"
    else:
        cfg.model_config.action_dim = act_space.shape[0]
        cfg.model_config.action_type = "continuous"

    print(f"[patch] Patched env space: state_dim={cfg.model_config.state_dim}, "
          f"action_dim={cfg.model_config.action_dim}, type={cfg.model_config.action_type}")
    
          
class DummyData:
    def __init__(self):
        self.buffer = []
        self.capacity = 100000

    def add(self, s, a, r, next_s, done):
        self.buffer.append((s, a, r, next_s, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.buffer[i] for i in idx])
        return dict(
            s1=np.array(s, dtype=np.float32),
            a1=np.array(a, dtype=np.int64).reshape(-1, 1),
            reward=np.array(r, dtype=np.float32).reshape(-1, 1),
            s2=np.array(s2, dtype=np.float32),
            dsc=np.array(d, dtype=np.float32).reshape(-1, 1),
        )


def main():

    command_args = {
        'model.model_name': 'dqn',
        'env.external.benchmark_name': 'gym',
        'env.external.env_name': 'CartPole-v1',
        'train.device': 'cuda',
        'train.total_train_steps': 2000000,
        'train.batch_size': 128,
    }

    config = make_config(command_args)


    env = gym_env(env_name='CartPole-v1', config = config)
    patch_config_for_discrete_env(config , env)
    data = DummyData()

    agent = make_agent(config = config, env = env, data = data)
    evaluator = bm_eval(agent = agent, env = env, config = config)
    trainer = Trainer(agent = agent, train_data = data, config = config, env = env, evaluator = evaluator)

    wandb.init(
        project="d2c",
        name="CartPole-v1_dqn",
        config=command_args,
        mode="online",
    )

    obs = env.reset()
    total_steps = int(config.model_config.train.total_train_steps)
    batch_size = int(config.model_config.train.batch_size)
    learning_starts = 10000       
    train_freq = 10               
    target_update_freq = 500      
    episode_rewards = []
    ep_reward = 0

    for step in range(total_steps):
        obs_np = np.array(obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]

        agent.update_epsilon(step, total_steps)
        if np.random.rand() < agent.epsilon:
            action = env.action_space.sample()
        else:
            action = agent._sampler_policy(obs_np)

        next_obs, reward, done, info = env.step(action)
        data.add(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward

        if step > learning_starts and step % train_freq == 0:
            batch = data.sample_batch(batch_size)
            if batch is not None:
                q_info = agent._optimize_step(batch)
                wandb.log({
                    "train/Q_loss": q_info["Q_loss"],
                    "train/average_q_pred": q_info["average_q_pred"],
                    "train/average_q_target": q_info["average_q_target"],
                    "train/global_step": step,
                    "train/epsilon": agent.epsilon,
                })

        if done:
            episode_rewards.append(ep_reward)
            mean_reward = np.mean(episode_rewards[-10:])
            wandb.log({
                "eval/main-episode_mean_score": ep_reward,
                "eval/main-episode_mean_score_smooth": mean_reward,
            })
            obs = env.reset()
            ep_reward = 0

        if step % 1000 == 0:
            print(f"[DQN] Step {step}/{total_steps}, Buffer: {len(data.buffer)}, ε={agent.epsilon:.3f}")

    wandb.finish()


if __name__ == "__main__":
    main()
