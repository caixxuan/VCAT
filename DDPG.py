import random, joblib
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch,os
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.model_id = 0
        self.model_path = '/home/oem/SafeBench/highway_simulation/'
        self.name = 'ddpg'

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(1)
        return action.item()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def load_model(self):
        filepath = os.path.join(self.model_path, f'model.ddpg.{self.model_id}.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
        try:
            result = joblib.load('result_ddpg_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result
    
    def save_model(self,return_list, ego_records_list):
        states = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.ddpg.{self.model_id}.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result['ego_records'] = ego_records_list
        joblib.dump(result, 'result_ddpg_scenario#6.pkl')
        


actor_lr = 8e-4
critic_lr = 8e-3
num_episodes = 2000
hidden_dim = 64
gamma = 0.95
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.2  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    "lanes_count": 2,
    "initial_spacing": 2,
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
    "on_road_reward": 0,
})
env.unwrapped.configure(env.unwrapped.config)
seed = 0
env.reset(seed=seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed=seed)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = 8
action_dim = 1
action_bound = 3  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
result = agent.load_model()
num_episodes = 2000 - len(result['episode'])

return_list, ego_records_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size,\
                                                                result["episode_reward"],result["ego_records"])
agent.save_model(return_list,ego_records_list)

episodes_list = result['episode']
result = agent.load_model()
plt.plot(result['episode'], result['episode_reward'])
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()

