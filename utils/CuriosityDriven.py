import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VictimValueSurrogate(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VictimValueSurrogate, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        self.state_features = x
        x = self.fc2(x)
        return x

def td_learning(surr_model, states, rewards, next_states, dones, gamma, optimizer, device):
    """
    使用Temporal Difference学习更新代理网络参数
    :param surr_model: 代理网络模型
    :param states: 当前状态
    :param rewards: 立即奖励
    :param next_states: 下一状态
    :param gamma: 折扣因子
    :param optimizer: 优化器
    :param device: 运算设备
    """
    # 预测当前状态的价值
    V_curr = surr_model(states)
    # 预测下一状态的价值
    V_next = surr_model(next_states)
    V_next_target = rewards + gamma * V_next * (1-dones)  # TD Target
    # 计算损失
    loss = torch.mean(F.mse_loss(V_next_target.detach(), V_curr))
    # 反向传播并优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(),V_curr,V_next

# RND FRAMEWORK
class TargetNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, RND_outsize):
        super(TargetNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, RND_outsize)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PredictorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, RND_outsize):
        super(PredictorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, RND_outsize)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RND:
    def __init__(self, input_size, hidden_size, RND_outsize, learning_rate, device):
        super(RND, self).__init__()
        self.target_net=TargetNetwork(input_size, hidden_size, RND_outsize).to(device)
        self.predictor_net=PredictorNetwork(input_size, hidden_size, RND_outsize).to(device)
        # 固定目标网络参数
        for param in self.target_net.parameters():
            param.requires_grad = False
        # 设置优化器
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=learning_rate)

    def compute_intrinsic_reward(self, state_features):
        # 假设state_features是来自受害代理价值网络的隐藏层输出
        with torch.no_grad():
            target_output = self.target_net(state_features)
        predictor_output = self.predictor_net(state_features)
        # 计算预测误差，作为内在奖励
        intrinsic_reward_each = torch.square(predictor_output - target_output)
        intrinsic_loss = intrinsic_reward_each.mean().unsqueeze(0)
        return intrinsic_reward_each, intrinsic_loss

    def train_predictor_network(self, state_features):
        self.optimizer.zero_grad()
        intrinsic_reward_each, intrinsic_loss = self.compute_intrinsic_reward(state_features)
        intrinsic_loss.backward()
        self.optimizer.step()
        return intrinsic_reward_each

class CD():
    def __init__(self, input_size, hidden_size, surr_outputsize, RND_outsize):
        super(CD, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 超参数定义
        lr = 0.001
        self.gamma = 0.99  # 折扣因子
        self.surr_model = VictimValueSurrogate(input_size, hidden_size, surr_outputsize).to(self.device)
        self.optimizer = optim.Adam(self.surr_model.parameters(), lr=lr)
        self.rnd = RND(hidden_size, 2*hidden_size, RND_outsize, lr, self.device)

    def surrogate_update(self,states,rewards,next_states,dones):
        # 执行TD学习更新
        states = states.clone().detach().float().to(self.device)
        rewards = rewards.clone().detach().float().to(self.device)
        next_states = next_states.clone().detach().float().to(self.device)
        self.loss_surrogate,V_curr,V_next = td_learning(self.surr_model, states, rewards, next_states, dones, \
                                          self.gamma, self.optimizer, self.device)
        return V_curr,V_next

    def RND_update(self):
        intrinsic_reward = self.rnd.train_predictor_network(self.surr_model.state_features.to(self.device))
        return intrinsic_reward
    
    def set_mode(self, mode):
        if mode == 'train':
            self.surr_model.train()
            self.rnd.predictor_net.train()
        elif mode == 'eval':
            self.surr_model.eval()
            self.rnd.predictor_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')