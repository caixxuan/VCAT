''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:01:35
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>

    This file implements the method proposed in paper:
        Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method
        <https://arxiv.org/pdf/2003.01197.pdf>
'''

# Learning-to-Collide不能直接用在highway-env环境，还需要进一步的修改，暂时还是用CARLA仿真

import os, joblib, rl_utils, random
import matplotlib.pyplot as plt
import numpy as np
import torch
from fnmatch import fnmatch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym

def normalize_routes(routes):
    mean_x = np.mean(routes[:, 0:1])
    max_x = np.max(np.abs(routes[:, 0:1]))
    x_1_2 = (routes[:, 0:1] - mean_x) / (max_x+1e-8)

    mean_y = np.mean(routes[:, 1:2])
    max_y = np.max(np.abs(routes[:, 1:2]))
    y_1_2 = (routes[:, 1:2] - mean_y) / (max_y+1e-8)

    route = np.concatenate([x_1_2, y_1_2], axis=0)
    return route


class IndependantModel(nn.Module):
    def __init__(self, num_waypoint=20):
        super(IndependantModel, self).__init__()
        input_size = num_waypoint*2 + 1
        hidden_size_1 = 64

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        self.d_os = 1

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os*2))
        self.fc_action_b = nn.Sequential(nn.Linear(1+hidden_size_1, self.b_os*2))
        self.fc_action_c = nn.Sequential(nn.Linear(1+1+hidden_size_1, self.c_os*2))
        self.fc_action_d = nn.Sequential(nn.Linear(1+1+1+hidden_size_1, self.d_os*2))

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = torch.randn(mu.size()).to(self.device)
        action = (mu + sigma*eps)
        return action, mu, sigma

    def forward(self, x, determinstic):
        # p(s)
        s = self.fc_input(x)
        s = self.relu(s)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s) 
        normal_b = self.fc_action_b(s)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        normal_c = self.fc_action_c(s)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        normal_d = self.fc_action_d(s)
        action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        # concate
        action = torch.cat((action_a, action_b, action_c, action_d), dim=1) # [B, 4]
        mu = torch.cat((mu_a, mu_b, mu_c, mu_d), dim=1)                     # [B, 4]
        sigma = torch.cat((sigma_a, sigma_b, sigma_c, sigma_d), dim=1)      # [B, 4]
        return mu, sigma, action


class AutoregressiveModel(nn.Module):
    def __init__(self, num_waypoint=30, standard_action_dim=True):
        super(AutoregressiveModel, self).__init__()
        self.standard_action_dim = standard_action_dim
        input_size = num_waypoint*2 + 1
        hidden_size_1 = 64

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        if self.standard_action_dim:
            self.d_os = 1

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os*2))
        self.fc_action_b = nn.Sequential(nn.Linear(1+hidden_size_1, self.b_os*2))
        self.fc_action_c = nn.Sequential(nn.Linear(1+1+hidden_size_1, self.c_os*2))
        if self.standard_action_dim:
            self.fc_action_d = nn.Sequential(nn.Linear(1+1+1+hidden_size_1, self.d_os*2))
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = torch.randn(mu.size()).to(self.device)
        action = mu + sigma * eps
        return action, mu, sigma

    def forward(self, x, determinstic):
        # p(s)
        s = self.fc_input(x)
        s = self.relu(s)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s)
        state_sample_a = torch.cat((s, mu_a), dim=1) if determinstic else torch.cat((s, action_a), dim=1) 
        normal_b = self.fc_action_b(state_sample_a)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, mu_a, mu_b), dim=1) if determinstic else torch.cat((s, action_a, action_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        if self.standard_action_dim:
            state_sample_a_b_c = torch.cat((s, mu_a, mu_b, mu_c), dim=1) if determinstic else torch.cat((s, action_a, action_b, action_c), dim=1)
            normal_d = self.fc_action_d(state_sample_a_b_c)
            action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        # concate
        if self.standard_action_dim:
            action = torch.cat((action_a, action_b, action_c, action_d), dim=1) # [B, 4]
            mu = torch.cat((mu_a, mu_b, mu_c, mu_d), dim=1)                     # [B, 4]
            sigma = torch.cat((sigma_a, sigma_b, sigma_c, sigma_d), dim=1)+1e-8     # [B, 4]
        else:
            action = torch.cat((action_a, action_b, action_c), dim=1)           # [B, 3]
            mu = torch.cat((mu_a, mu_b, mu_c), dim=1)                           # [B, 3]
            sigma = torch.cat((sigma_a, sigma_b, sigma_c), dim=1)+1e-8               # [B, 3]
        return mu_a, sigma_a, action_a


class REINFORCE():
    name = 'reinforce'
    type = 'init_state'

    def __init__(self,device):
        self.num_waypoint = 30
        self.continue_episode = 0
        self.num_scenario = 1
        self.batch_size = 64
        self.device = device
        self.model_id = 0
        self.model_path = '/home/oem/SafeBench/highway_simulation/'
        self.name = 'lc'
        self.lr = 8.0e-4
        self.entropy_weight = 5.0e-3

        self.standard_action_dim = True
        self.model = AutoregressiveModel(self.num_waypoint).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def update(self, transition_dict, additional_info=None):
        # get episode reward
        episode_reward = transition_dict['rewards']
        log_prob = additional_info['log_prob']
        entropy = additional_info['entropy']
        
        episode_reward = torch.tensor(episode_reward, dtype=torch.float32).to(self.device)
        episode_reward = -episode_reward/100 # objective is to minimize the reward: greater reward of ego, greater loss of scenario, caixuan
        # we only have one step
        loss = log_prob * episode_reward - entropy * self.entropy_weight
        loss = loss.mean(dim=0)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def proceess_init_state(self, state):
        processed_state_list = []

        route = state['route']
        target_speed = state['target_speed'] / 10.0

        index = np.linspace(1, len(route) - 1, self.num_waypoint).tolist()
        index = [int(i) for i in index]
        route_norm = normalize_routes(route[index])[:, 0] # [num_waypoint*2]
        processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype('float32')
        processed_state_list.append(processed_state)
        
        processed_state_list = np.stack(processed_state_list, axis=0)
        return processed_state_list

    def take_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def take_init_action(self, state, state_, deterministic=False):
        # the state should be a sequence of route waypoints
        processed_state = self.proceess_init_state(state_)
        processed_state = torch.from_numpy(processed_state).to(self.device)

        mu, sigma, action = self.model.forward(processed_state, deterministic)

        # calculate the probability that this distribution outputs this action
        action_dist = Normal(mu, sigma)
        log_prob = action_dist.log_prob(action).sum(dim=1) # [B]

        # calculate the entropy
        action_entropy = 0.5*(2 * np.pi * sigma**2).log() + 0.5
        entropy = action_entropy.sum(dim=1) # [B]

        # clip the action to [-1, 1]
        action = np.clip(action.detach().cpu(), -1.0, 1.0)
        additional_info = {'log_prob': log_prob, 'entropy': entropy}
        return [action[-1][-1]], additional_info

    def load_model(self):
        filepath = os.path.join(self.model_path, f'model.lc.{self.model_id}.torch')
        self.model = AutoregressiveModel(self.num_waypoint, self.standard_action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
        try:
            result = joblib.load('result_lc_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result

    def save_model(self, return_list, ego_records_list):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        filepath = os.path.join(self.model_path, f'model.lc.{self.model_id}.torch')
        with open(filepath, 'wb+') as f:
            torch.save({'parameters': self.model.state_dict()}, f)
        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result['ego_records'] = ego_records_list
        joblib.dump(result, 'result_lc_scenario#6.pkl')


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
state_dim = 8
action_dim = 1
action_bound = 3  # 动作最大值
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = REINFORCE(device)
result = agent.load_model()
num_episodes = 2000 - len(result['episode'])

return_list, ego_records_list = rl_utils.train_on_policy_agent_lc(env, agent, num_episodes, result["episode_reward"],result["ego_records"])
agent.save_model(return_list, ego_records_list)

result = agent.load_model()
plt.plot(result['episode'], result['episode_reward'])
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()

