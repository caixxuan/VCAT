import random, joblib
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch,os,copy
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from fnmatch import fnmatch

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, action_bound=3):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)
        self.tanh = nn.Tanh()
        self.action_bound = action_bound

    def forward(self, x):
        x = self.network(x)
        x = self.tanh(x) * self.action_bound
        return x


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class TD3():
    name = 'TD3'
    type = 'offpolicy'

    def __init__(self,device):

        self.lr = 8.0e-4
        self.state_dim = 8
        self.action_dim = 1
        self.hidden_size = 128
        self.update_iteration = 5
        self.gamma = 0.95
        self.tau = 8.0e-3
        self.update_interval = 2
        self.action_lim = 3
        self.target_noise = 0.5
        self.target_noise_clip = 0.5
        self.explore_noise = 0.5
        self.device = device

        self.model_id = 0
        self.model_path = '/home/oem/SafeBench/highway_simulation/'
        self.name = 'td3'

        # aka critic
        self.q_funcs = DoubleQFunc(self.state_dim, self.action_dim, hidden_size=self.hidden_size).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(self.state_dim, self.action_dim, hidden_size=self.hidden_size).to(self.device)
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self._update_counter = 0
        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.q_funcs.train()
            self.target_q_funcs.train()
            self.policy.train()
            self.target_policy.train()
        elif mode == 'eval':
            self.q_funcs.eval()
            self.target_q_funcs.eval()
            self.policy.eval()
            self.target_policy.eval()
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

    def take_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return action.cpu().item()

    def update_target(self):
        # moving average update of target networks
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch = self.target_policy(nextstate_batch)
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy(self, state_batch):
        action_batch = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    def update(self, transition_dict):
        q1_loss, q2_loss, pi_loss = 0, 0, None
        for _ in range(self.update_iteration):
            bn_s = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
            bn_a = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
            bn_r = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            bn_s_ = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            bn_d = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(bn_s, bn_a, bn_r, bn_s_, bn_d)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()
            self._update_counter += 1

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()

            if self._update_counter % self.update_interval == 0:
                if not pi_loss:
                    pi_loss = 0
                # update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step = self.update_policy(bn_s)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
                # update target policy and q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        return q1_loss, q2_loss, pi_loss

    def save_model(self, return_list, ego_records_list):
        states = {
            'q_funcs': self.q_funcs.state_dict(),
            'target_q_funcs': self.target_q_funcs.state_dict(),
            'policy': self.policy.state_dict(),
            'target_policy': self.target_policy.state_dict(),
        }
        filepath = os.path.join(self.model_path, f'model.td3.{self.model_id}.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        
        episodes_list = list(range(len(return_list)))
        result['episode'] = episodes_list
        result["episode_reward"] = return_list
        result['ego_records'] = ego_records_list
        joblib.dump(result, 'result_td3_scenario#6.pkl')

    def load_model(self):
        filepath = os.path.join(self.model_path, f'model.td3.{self.model_id}.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.q_funcs.load_state_dict(checkpoint['q_funcs'])
            self.target_q_funcs.load_state_dict(checkpoint['target_q_funcs'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.target_policy.load_state_dict(checkpoint['target_policy'])
        try:
            result = joblib.load('result_td3_scenario#6.pkl')
        except:
            result = {'episode':[],'episode_reward':[],'ego_records':[]}
        return result


buffer_size = 10000
minimal_size = 1000
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = gym.make("inverse6-env-cx-v0", render_mode='rgb_array')
env.unwrapped.config.update({
    "duration": 60,
    "controlled_vehicles": 1,  # 受控车辆数量
    "destination": "o1",
    "vehicles_count": 2,  # 交通车辆数量
    "initial_vehicle_count": 2,  # 初始车辆数量
    "spawn_probability": 0,  # 新车辆生成概率，设为0确保没有额外车辆生成
    "offroad_terminal": True,  # 车辆离开道路则终止
    "manual_control": False,  # 是否手动控制
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
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = TD3(device)
result = agent.load_model()
num_episodes = 2000 - len(result['episode'])
# return_list, ego_records_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, \
#                                                                 batch_size, result["episode_reward"], result["ego_records"])
# agent.save_model(return_list,ego_records_list)

episodes_list = result['episode']
result = agent.load_model()
plt.plot(result['episode'], result['episode_reward'])
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.show()
