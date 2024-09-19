import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils,os
from fnmatch import fnmatch
import joblib

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 3.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.model_id = 0
        self.model_path = '/home/oem/SafeBench/highway_simulation/'
        self.name = 'ppo'
        self.continue_episode = 0

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def load_model(self):
        filepath = os.path.join(self.model_path, f'model.ppo.{self.model_id}.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
        try:
            result = joblib.load('result_ppo_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result
    
    def save_model(self,return_list,ego_records_list):
        states = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.ppo.{self.model_id}.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result['ego_records'] = ego_records_list
        joblib.dump(result, 'result_ppo_scenario#6.pkl')
        

actor_lr = 8.0e-4
critic_lr = 8.0e-3

hidden_dim = 128
gamma = 0.95
lmbda = 0.9
epochs = 5
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


env = gym.make("inverse6-env-cx-v0", render_mode='rgb_array')
env.unwrapped.config.update({
    "duration": 60,
    "controlled_vehicles": 1,  # 受控车辆数量
    "destination": "o1",
    "vehicles_count": 1,  # 交通车辆数量
    "initial_vehicle_count": 1,  # 初始车辆数量
    "spawn_probability": 0,  # 新车辆生成概率，设为0确保没有额外车辆生成
    "offroad_terminal": True,  # 车辆离开道路则终止
    "action": {
        "type": "ContinuousAction",  # 动作类型
    },
    "observation": {
        "type": "Kinematics",  # 观察类型
        "features": ["x", "y", "vx", "vy"],
        "normalize": True,
    },
    "other_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle',
    "policy_frequency": 2,  # 决策频率
    "simulation_frequency": 10,  # 模拟频率
    "collision_reward": 100,
    "high_speed_reward": -0.1,
    "arrived_reward": -2,
    "on_road_reward": 0.0,
})
env.unwrapped.configure(env.unwrapped.config)
seed = 0
env.reset(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed=seed)
state_dim = 8
action_dim = 1
action_bound = 3  # 动作最大值
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

result = agent.load_model()
num_episodes = 2000 - len(result['episode'])
return_list, ego_records_list = rl_utils.train_on_policy_agent(env, agent, num_episodes, result["episode_reward"],result["ego_records"])
agent.save_model(return_list,ego_records_list)

episodes_list = result['episode']
result = agent.load_model()
plt.plot(result['episode'], result['episode_reward'])
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()
