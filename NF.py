''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:26:29
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>

    This file implements the method proposed in paper:
        Multimodal Safety-Critical Scenarios Generation for Decision-Making Algorithms Evaluation
        <https://arxiv.org/pdf/2009.08311.pdf>
'''

import os, joblib, rl_utils, random
import numpy as np
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import gymnasium as gym

class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3_s = nn.Linear(n_hidden, n_output)
        self.fc3_t = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        hidden = F.relu(self.fc2(F.relu(self.fc1(x))))
        s = torch.tanh(self.fc3_s(hidden))
        t = self.fc3_t(hidden)
        return s, t


class ConditionalRealNVP(nn.Module):
    # Generator, condition_dim=self.state_dim, data_dim=self.action_dim
    def __init__(self, n_flows, condition_dim, data_dim, n_hidden):
        super(ConditionalRealNVP, self).__init__()
        self.n_flows = n_flows
        self.condition_dim = condition_dim

        # divide the data dimension by 1/2 to do the affine operation
        assert(data_dim % 2 == 0)
        self.n_half = int(data_dim/2)

        # build the network list
        self.NN = torch.nn.ModuleList()
        for k in range(n_flows):
            # the input of each layer should also contain the condition
            # self.n_half+self.condition_dim作为输入维度。
            # 这是因为在每个流中，在执行仿射变换之前，条件维度与数据的一半维度进行拼接（concatenation）。
            # 因此，该神经网络除了要处理原始数据的一半，还要处理条件数据。
            self.NN.append(MLP(self.n_half+self.condition_dim, self.n_half, n_hidden)) # 这个self.NN不会和后面的RealNVP冲突吗？？
        
    def forward(self, x, c):
        log_det_jacobian = 0
        for k in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]
            
            x_a_c = torch.cat([x_a, c], dim=1)
            s, t = self.NN[k](x_a_c)
            x_b = torch.exp(s)*x_b + t
            
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        
        return x, log_det_jacobian
        
    def inverse(self, z, c):
        # 反向操作是将变换后的数据（这里表示为z,(mean)）重新映射回原始数据空间的流程。在这个上下文中，z表示已经转换过的数据，而c代表了条件变量。
        for k in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]

            z_a_c = torch.cat([z_a, c], dim=1)

            s, t = self.NN[k](z_a_c) # 使用第k个神经网络获取尺度参数s和平移参数t, self.NN[k]调用了第k个子模型，传入数据z_a_c进行前向计算，分别输出尺度和平移参数。
            z_b = (z_b - t) / torch.exp(s) # 使用尺度和平移参数更新z_b。在RealNVP的反向过程中，需要对z_b执行反运算，即减去平移参数t，然后除以exp(s)（因为在正向中是乘以exp(s)）。
            z = torch.cat([z_a, z_b], dim=1)
        return z


# for prior model，flow-based）模型，用于生成模型任务，如样本生成或概率密度估计，从简单分布（如高斯分布）有效地生成复杂数据分布的样本。
class RealNVP(nn.Module):
    def __init__(self, n_flows, data_dim, n_hidden): 
        # n_flows：流的数量。在RealNVP模型中，数据会依次通过这些流进行转换，每个流都尝试捕捉并建模数据的不同特征和依赖。
        super(RealNVP, self).__init__()
        self.n_flows = n_flows

        # divide the data dimension by 1/2 to do the affine operation
        # data_dim：数据的维度。它是每个数据点的特征或维度数量。==self.action_dim，NPC的动作维度？
        # n_hidden：隐藏层的维度。这在构建每个流使用的神经网络时使用。
        assert(data_dim % 2 == 0) # 一半的数据会用于条件变换，另一半进行仿射变换（affine transformation）
        self.n_half = int(data_dim/2) 

        # build the network list
        self.NN = torch.nn.ModuleList() # 在RealNVP模型中，每个“流（flow）”都是通过具体的神经网络来实现的，这里将用它来存储所有流的网络 
        for k in range(n_flows):
            # the input of each layer should also contain the condition
            # 对于每个流，通过循环创建一个MLP（多层感知器）网络，并将其添加到NN列表中。
            # 这个MLP的输入和输出维度都是self.n_half，表示每个流网络仅处理数据的一半。
            # n_hidden是传递给MLP的另一个参数，表示隐藏层的大小。
            self.NN.append(MLP(self.n_half, self.n_half, n_hidden))
        
    def forward(self, x):
        log_det_jacobian = 0
        for k in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]

            s, t = self.NN[k](x_a)
            x_b = torch.exp(s)*x_b + t
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        
        return x, log_det_jacobian
        
    def inverse(self, z):
        for k in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]
            s, t = self.NN[k](z_a)
            z_b = (z_b - t) / torch.exp(s)
            z = torch.cat([z_a, z_b], dim=1)
        return z


class NormalizingFlow():
    name = 'nf'
    type = 'init_state'

    def __init__(self, device):
        self.num_waypoint = 31
        self.continue_episode = 0
        self.device = device
        self.model_id = 0
        self.model_path = '/home/oem/SafeBench/highway_simulation/'
        self.name = 'lc'
        self.use_prior = False # 使用先验与否

        self.lr = 8.0e-4
        self.batch_size = 128
        self.prior_lr = 8.0e-4

        self.prior_epochs = 10
        self.alpha = 0.5
        self.itr_per_train = 10

        self.state_dim = 63 # 路点数量？63 31*2+1
        self.action_dim = 4 # ？
        self.reward_dim = 4 # ？
        self.drop_threshold = 0.1
        self.n_flows = 4 # ？

        # latent space 创建多元正态（或高斯）分布的类。这个分布有两个参数——均值向量和协方差矩阵，均值向量，协方差矩阵及其精度矩阵（逆）和Cholesky分解（下三角）
        self.z = MultivariateNormal(torch.zeros(self.action_dim).to(self.device), torch.eye(self.action_dim).to(self.device)) # --p(z|x), simple distribution, 

        # prior model and generator
        self.prior_model = RealNVP(n_flows=self.n_flows, data_dim=self.action_dim, n_hidden=128).to(self.device)
        self.model = ConditionalRealNVP(n_flows=self.n_flows, condition_dim=self.state_dim, data_dim=self.action_dim, n_hidden=64).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_prior_model(self, prior_data):
        """ 
            Train the prior model using the data from the prior distribution.
            This function should be used seperately from the Safebench framework to train the prior model.
        """
        prior_data = torch.tensor(prior_data).to(self.device)
        # papre a data loader
        train_loader = torch.utils.data.DataLoader(prior_data, shuffle=True, batch_size=self.batch_size)
        self.prior_optimizer = optim.Adam(self.prior_model.parameters(), lr=self.prior_lr)
        self.prior_model.train()

        # train the model
        for epoch in range(self.prior_epochs):
            avg_loglikelihood = []
            for data in train_loader:
                sample_z, log_det_jacobian = self.prior_model(data)
                log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
                loglikelihood = -torch.mean(self.z.log_prob(sample_z)[:, None] + log_det_jacobian)
                self.prior_optimizer.zero_grad()
                loglikelihood.backward()
                self.prior_optimizer.step()
                avg_loglikelihood.append(loglikelihood.item())
            self.logger.log('[{}/{}] Prior training error: {}'.format(epoch, self.prior_epochs, np.mean(avg_loglikelihood)))
        self.save_prior_model()

    def prior_likelihood(self, actions):
        sample_z, log_det_jacobian = self.prior_model(actions)
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
        loglikelihood = self.z.log_prob(sample_z)[:, None] + log_det_jacobian
        prob = torch.exp(loglikelihood)
        return prob

    def flow_likelihood(self, actions, condition):
        sample_z, log_det_jacobian = self.model(actions, condition)
        # make sure the dimension is aligned, for action_dim > 2, the log_det is more than 1 dimension
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True) # 雅可比矩阵
        loglikelihood = self.z.log_prob(sample_z)[:, None] + log_det_jacobian
        return loglikelihood

    def prior_sample(self, sample_number=1000, sigma=1.0): # 没有用到？
        sampler = MultivariateNormal(torch.zeros(self.action_dim).to(self.device), sigma*torch.eye(self.action_dim).to(self.device))
        new_sampled_z = sampler.sample((sample_number,))

        self.prior_model.eval()
        with torch.no_grad():
            prior_flow = self.prior_model.inverse(new_sampled_z)
        return prior_flow.cpu().numpy()

    def flow_sample(self, state, sample_number=1000, sigma=1.0): 
        # use a new sampler, then we can control the sigma 
        sampler = MultivariateNormal(torch.zeros(self.action_dim).to(self.device), sigma*torch.eye(self.action_dim).to(self.device))
        new_sampled_z = sampler.sample((sample_number,))

        # condition should be repeated sample_number times
        condition = state.clone().detach().requires_grad_(True).to(self.device)
        condition = condition.repeat(sample_number, 1)

        self.model.eval()
        with torch.no_grad():
            action_flow = self.model.inverse(new_sampled_z, condition)
        return action_flow

    def normalize_routes(self, routes):
        mean_x = np.mean(routes[:, 0:1])
        max_x = np.max(np.abs(routes[:, 0:1]))
        x_1_2 = (routes[:, 0:1] - mean_x) / (max_x+1e-8)

        mean_y = np.mean(routes[:, 1:2])
        max_y = np.max(np.abs(routes[:, 1:2]))
        y_1_2 = (routes[:, 1:2] - mean_y) / (max_y+1e-8)

        route = np.concatenate([x_1_2, y_1_2], axis=0)
        return route

    def proceess_init_state(self, state):
        processed_state_list = []
        for i in range(len(state)):
            route = state[i]['route']
            target_speed = state[i]['target_speed'] / 10.0

            index = np.linspace(1, len(route) - 1, self.num_waypoint).tolist()#采样
            index = [int(i) for i in index]
            route_norm = self.normalize_routes(route[index])[:, 0] # [num_waypoint*2]
            processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype('float32')
            processed_state_list.append(processed_state)
            
        processed_state_list = np.stack(processed_state_list, axis=0)
        return processed_state_list

    def take_init_action(self, state, state_, deterministic=False):
        # the state should be a sequence of route waypoints + target_speed (31*2+1)
        processed_state = self.proceess_init_state([state_])
        processed_state = torch.from_numpy(processed_state).to(self.device)

        self.model.eval() # 确保模型在评估模式下运行，关闭Dropout和BatchNorm等影响模型表现的因素。
        with torch.no_grad(): # 将PyTorch的梯度计算暂时关闭，用于阻止Autograd（自动梯度）引擎从计算和存储梯度，从而提高内存效率并加速计算
            action_flow = self.flow_sample(processed_state,sigma=0.8)
            random_index = torch.randint(0, action_flow.size(0), (1,)).item()  # item() 获得Python数值
            action = action_flow[random_index][None]

        action_list = []
        for a_i in range(self.action_dim):
            action_list.append(action.cpu().numpy()[0, a_i])
        
        return action_list

    # train on batched data
    def update(self, additional_info=None):
        if len(additional_info['static_obs']) < self.batch_size:
            return
        
        self.model.train()
        # the buffer can be resued since we evaluate action-state every time
        for _ in range(self.itr_per_train):
            # get episode reward
            additional_info = self.sample(additional_info)
            state = np.array(additional_info['static_obs'])
            action = np.array(additional_info['ego_action'])
            episode_reward = np.array(additional_info['rewards'])
            
            processed_state = self.proceess_init_state(state) # caixuan 62 WP + 1 TS
            processed_state = torch.from_numpy(processed_state).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            episode_reward = torch.from_numpy(episode_reward)[None].t().to(self.device)

            loglikelihood = self.flow_likelihood(action, processed_state) # log P_x(x|\theta)，
            prior_prob = self.prior_likelihood(action) if self.use_prior else 0 # P_x(x|\theta),先验分布,论文对应q(x)
            assert loglikelihood.shape == episode_reward.shape

            # this term is actually the log-likelihood weighted by reward
            loss_r = -(loglikelihood * (torch.exp(-episode_reward/100) + self.alpha * prior_prob)).mean()
            self.optimizer.zero_grad()
            loss_r.backward()
            self.optimizer.step()

    def sample(self, additional_info):
        random_indices = random.sample(range(len(list(additional_info.values())[0])), self.batch_size)
        # 对additional_info的每个列表进行采样
        sampled_dict = {}
        for key, value_list in additional_info.items():
            sampled_dict[key] = [value_list[i] for i in random_indices]
        return sampled_dict

    def load_model(self):
        filepath = os.path.join(self.model_path, f'model.nf.{self.model_id}.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
        try:
            result = joblib.load('result_nf_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result

    def save_model(self, return_list, ego_records_list):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        filepath = os.path.join(self.model_path, f'model.nf.{self.model_id}.torch')
        with open(filepath, 'wb+') as f:
            torch.save({'parameters': self.model.state_dict()}, f)
        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result['ego_records'] = ego_records_list
        joblib.dump(result, 'result_nf_scenario#6.pkl')

    def save_prior_model(self):
        states = {'parameters': self.prior_model.state_dict()}
        model_filename = os.path.join(self.model_path, 'nf.prior.'+str(self.model_id)+'.pt')
        with open(model_filename, 'wb+') as f:
            torch.save(states, f)
            self.logger.log(f'>> Save prior model of nf')

    def load_prior_model(self):
        model_filename = os.path.join(self.model_path, 'nf.prior.'+str(self.model_id)+'.pt')
        self.logger.log(f'>> Loading nf model from {model_filename}')
        if os.path.isfile(model_filename):
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.prior_model.load_state_dict(checkpoint['parameters'])
        else:
            self.logger.log(f'>> Fail to find nf prior model from {model_filename}', color='yellow')


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
    "manual_control": False,
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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = NormalizingFlow(device)
result = agent.load_model()
num_episodes = 2000 - len(result['episode'])

# return_list, ego_records_list = rl_utils.train_on_policy_agent_nf(env, agent, num_episodes, result["episode_reward"],result["ego_records"])
# agent.save_model(return_list, ego_records_list)

result = agent.load_model()
plt.plot(result['episode'], result['episode_reward'])
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()
