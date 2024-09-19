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

class MC:
    def __init__(self, action_bound):
        self.action_bound = action_bound
        self.action_once = np.random.uniform(-self.action_bound, self.action_bound)

    def take_action(self, state):
        action = self.action_once
        return action
    def update(self, transition_dict):
        self.action_once = np.random.uniform(-self.action_bound, self.action_bound)

    def load_model(self):
        try:
            result = joblib.load('result_MC_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result

    def save_model(self, return_list, ego_records_list):        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result["ego_records"] = ego_records_list
        joblib.dump(result, 'result_MC_scenario#6.pkl')
        

env = gym.make("inverse6-env-cx-v0", render_mode='rgb_array')
env.unwrapped.config.update({
    "duration": 70,
    "controlled_vehicles": 1,  # 受控车辆数量
    "destination": "o1",
    "vehicles_count": 1,  # 交通车辆数量
    "initial_vehicle_count": 1,  # 初始车辆数量
    "spawn_probability": 0,  # 新车辆生成概率，设为0确保没有额外车辆生成
    "offroad_terminal": True,  # 车辆离开道路则终止
    "manual_control":False,
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
    "high_speed_reward": 0,
    "arrived_reward": 0.0,
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
action_bound = 2  # 动作最大值
agent = MC(action_bound)

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
