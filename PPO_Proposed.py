''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:17:00
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/gouxiangchen/ac-ppo>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os,rl_utils
import random
import numpy as np
import torch, joblib
import torch.nn as nn
from fnmatch import fnmatch
from torch.distributions import Normal
import torch.nn.functional as F
import gymnasium as gym
from utils import CuriosityDriven, rl_utils_ppo
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(PolicyNetwork, self).__init__()
        hidden_dim = 128
        self.action_bound = action_bound
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc_mu = torch.nn.Linear(int(hidden_dim/2), action_dim)
        self.fc_std = torch.nn.Linear(int(hidden_dim/2), action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.action_bound * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 1e-10
        return mu, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        hidden_dim = 128
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = torch.nn.Linear(int(hidden_dim/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPO_CD():
    name = 'PPO_CD'
    type = 'onpolicy'

    def __init__(self,state_dim, action_dim, action_bound):

        self.gamma = 0.95
        self.policy_lr = 8.0e-4
        self.value_lr = 8.0e-3
        self.train_iteration = 10
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.clip_epsilon = 0.2
        self.batch_size = 128
        self.lmbda = 0.9
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_id = 0
        self.model_path = '/home/oem/SafeBench/highway_simulation/'
        self.name = 'ppo_cd'

        self.policy = PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim, action_bound=self.action_bound).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value = ValueNetwork(state_dim=self.state_dim).to(self.device)
        self.value_ins = ValueNetwork(state_dim=self.state_dim).to(self.device)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.value_ins_optim = torch.optim.Adam(self.value_ins.parameters(), lr=self.value_lr)
        self.cd = CuriosityDriven.CD(self.state_dim, 32, 1, 1)

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
            self.value_ins.train()
            self.cd.set_mode('train')
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
            self.value_ins.eval()
            self.cd.set_mode('eval')
        else:
            raise ValueError(f'Unknown mode {mode}')

    def info_process(self, infos):
        info_batch = np.stack([i_i['actor_info'] for i_i in infos], axis=0)
        info_batch = info_batch.reshape(info_batch.shape[0], -1)
        return info_batch

    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_in = {}
        return [None] * num_scenario, additional_in

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mu, sigma = self.policy(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = action.clamp(-self.action_bound, self.action_bound)
        return action.item()
    
    def update(self, transition_dict):
        bn_s = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        bn_a = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        bn_r = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        bn_s_ = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        bn_d = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)

        if torch.isnan(bn_s).any() or torch.isinf(bn_s).any():
            return
        
        # bn_r = (bn_r+8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练

        rewards_attacker = bn_r
        # rewards_attacker = torch.where(bn_r > 60, torch.ones_like(bn_r), -torch.ones_like(bn_r))
        td_target = rewards_attacker + self.gamma * self.value(bn_s_)*(1-bn_d)
        td_delta = td_target - self.value(bn_s)
        advantage_attack = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.policy(bn_s)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(bn_a)


        # Curiosity-Driven ------------------------------------------------------------------
        # rewards_env = torch.where(bn_r > 60, -torch.ones_like(bn_r), torch.ones_like(bn_r)) #R_v=R_joint(s_t,a(a_attack,a_victim)),根据情况更改！
        rewards_env = -bn_r
        V_victim_cur, V_victim_next = self.cd.surrogate_update(bn_s,rewards_env,bn_s_,bn_d)
        reward_intrinsic = self.cd.RND_update()
        td_delta_victim = rewards_env+self.gamma*V_victim_next*(1-bn_d)-V_victim_cur
        advantage_victim = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta_victim.cpu()).to(self.device)
        td_delta_attack_ins = reward_intrinsic+self.gamma*self.value_ins(bn_s_)*(1-bn_d)-self.value_ins(bn_s)
        advantage_attack_ins = rl_utils_ppo.compute_advantage(self.gamma, self.lmbda, td_delta_attack_ins.cpu()).to(self.device)
        lambda_ = 100 #the degree of exploration, 0.1?
        advantage_attack = advantage_attack+lambda_*advantage_attack_ins
        #------------------------------------------------------------------------------------
        

        # start to train, use gradient descent without batch size
        for K in range(self.train_iteration):
            mu, std = self.policy(bn_s)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(bn_a)
            ratio = torch.exp(log_probs - old_log_probs)

            # L1 = ratio * advantage
            # L2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
            # loss = torch.min(L1, L2)

            # advantage_attack = (advantage_attack-advantage_attack.mean())/(advantage_attack.std()+1e-10)# 归一化？
            # advantage_victim = (advantage_victim-advantage_victim.mean())/(advantage_victim.std()+1e-10)
            L1 = torch.min(torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage_attack, ratio*advantage_attack)
            L2 = torch.min(torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage_victim, ratio*advantage_victim)
            policy_loss = -torch.mean(L1-L2) #加负号？

            value_loss = torch.mean(F.mse_loss(self.value(bn_s), td_target.detach()))
            value_ins_loss = torch.mean(F.mse_loss(self.value_ins(bn_s),reward_intrinsic.detach()+self.gamma*self.value_ins(bn_s_)*(1-bn_d)))

            self.optim.zero_grad()
            self.value_optim.zero_grad()
            self.value_ins_optim.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            value_ins_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.value_ins.parameters(), max_norm=1.0)
            self.optim.step()
            self.value_optim.step()
            self.value_ins_optim.step()

        # print(f'Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, Value Ins Loss: {value_ins_loss.item()}')

    def load_model(self):
        filepath = os.path.join(self.model_path, f'model.ppo_cd.{self.model_id}.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.value_ins.load_state_dict(checkpoint['value_ins'])
            self.cd.surr_model.load_state_dict(checkpoint['surr_net'])
            self.cd.rnd.target_net.load_state_dict(checkpoint['RND_tar'])
            self.cd.rnd.predictor_net.load_state_dict(checkpoint['RND_pre'])
        try:
            result = joblib.load('result_ppo_cd_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result

    def save_model(self, return_list,ego_records_list):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'value_ins': self.value_ins.state_dict(),
            'surr_net': self.cd.surr_model.state_dict(),
            'RND_tar': self.cd.rnd.target_net.state_dict(),
            'RND_pre': self.cd.rnd.predictor_net.state_dict()
        }
        filepath = os.path.join(self.model_path, f'model.ppo_cd.{self.model_id}.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result["ego_records"] = ego_records_list
        joblib.dump(result, 'result_ppo_cd_scenario#6.pkl')


env = gym.make("inverse6-env-cx-v0", render_mode='rgb_array')
env.unwrapped.config.update({
    "duration": 60,
    "controlled_vehicles": 1,  # 受控车辆数量
    "destination": "o1",
    "vehicles_count": 2,  # 交通车辆数量
    "initial_vehicle_count": 2,  # 初始车辆数量
    "spawn_probability": 0,  # 新车辆生成概率，设为0确保没有额外车辆生成
    "offroad_terminal": True,  # 车辆离开道路则终止
    'manual_control': False,
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
agent = PPO_CD(state_dim,action_dim,action_bound)

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
