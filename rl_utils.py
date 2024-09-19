import gymnasium as gym
from matplotlib import pyplot as plt
# %matplotlib inline

from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes, return_list, ego_records_list):
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                state = state_select(state,env)
                state = state[:2,:].ravel()
                done = False
                truncated = False
                ego_records_tmp = []
                # state_ = get_wp(env)
                # action = agent.take_init_action(state,state_)
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step([action,0])
                    next_state = state_select(next_state,env)
                    acceleration, steering = get_action_of_sut(env) #注意控制的车是NPC车，交通车是SUT
                    ego_records_tmp.append([acceleration,steering])
                    next_state = next_state[:2,:].ravel()
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    # additional_info['static_obs'].append(state_)
                    # additional_info['ego_action'].append(action)
                    # additional_info['rewards'].append(reward)
                    state = next_state
                    episode_return += reward
                    # env.render()
                return_list.append(-episode_return)
                if episode_return > 50: # only save once collision
                    ego_records_list.append(ego_records_tmp)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if (i_episode+1) % 20 == 0:
                    agent.save_model(return_list, ego_records_list)
    return return_list, ego_records_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, return_list, ego_records_list):
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, info = env.reset()
                state = state_select(state,env) #只有使用第6个场景的时候才需要用
                state = state[:2,:].ravel()
                done = False
                truncated = False
                ego_records_tmp = []
                while not done and not truncated:
                    action = agent.take_action(state)
                    agent.i_episode = i_episode
                    next_state, reward, done, truncated, info = env.step([action,0])
                    next_state = state_select(next_state,env)
                    acceleration, steering = get_action_of_sut(env) #注意控制的车是NPC车，交通车是SUT
                    ego_records_tmp.append([acceleration,steering])
                    next_state = next_state[:2,:].ravel()
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        # print(state,action,next_state,reward,done,_)
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                    # env.render()
                return_list.append(-episode_return)
                if episode_return > 50: # only save once collision
                    ego_records_list.append(ego_records_tmp)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if (i_episode+1) % 50 == 0:
                    agent.save_model(return_list,ego_records_list)
    return return_list, ego_records_list

def train_on_policy_agent_lc(env, agent, num_episodes, return_list, ego_records_list):
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                state = state_select(state,env)
                state = state[:2,:].ravel()
                done = False
                truncated = False
                ego_records_tmp = []
                state_ = get_wp(env)
                action, additional_info = agent.take_init_action(state,state_)
                while not done and not truncated:
                    next_state, reward, done, truncated, info = env.step([action[-1],0])
                    next_state = state_select(next_state,env)
                    acceleration, steering = get_action_of_sut(env) #注意控制的车是NPC车，交通车是SUT
                    ego_records_tmp.append([acceleration,steering])
                    next_state = next_state[:2,:].ravel()
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    # additional_info['static_obs'].append(state_)
                    # additional_info['ego_action'].append(action)
                    # additional_info['rewards'].append(reward)
                    state = next_state
                    episode_return += reward
                    # env.render()
                return_list.append(-episode_return)
                if episode_return > 50: # only save once collision
                    ego_records_list.append(ego_records_tmp)
                agent.update(transition_dict, additional_info)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if (i_episode+1) % 50 == 0:
                    agent.save_model(return_list, ego_records_list)
    return return_list, ego_records_list

def train_on_policy_agent_nf(env, agent, num_episodes, return_list, ego_records_list):
    additional_info = {'ego_action':[],'static_obs':[],'rewards':[]}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset()
                state = state_select(state,env)
                state = state[:2,:].ravel()
                done = False
                truncated = False
                ego_records_tmp = []
                state_ = get_wp(env)
                action = agent.take_init_action(state,state_)
                while not done and not truncated:
                    next_state, reward, done, truncated, info = env.step([action[-1],0])
                    next_state = state_select(next_state,env)
                    acceleration, steering = get_action_of_sut(env) #注意控制的车是NPC车，交通车是SUT
                    ego_records_tmp.append([acceleration,steering])
                    next_state = next_state[:2,:].ravel()
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    additional_info['static_obs'].append(state_)
                    additional_info['ego_action'].append(action)
                    additional_info['rewards'].append(reward)
                    state = next_state
                    episode_return += reward
                    # env.render()
                return_list.append(-episode_return)
                if episode_return > 50: # only save once collision
                    ego_records_list.append(ego_records_tmp)
                agent.update(additional_info)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if (i_episode+1) % 20 == 0:
                    agent.save_model(return_list, ego_records_list)
    return return_list, ego_records_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def get_action_of_sut(env):
    vehicles = env.unwrapped.road.vehicles
    ego_vehicle = env.unwrapped.vehicle
    sut = [vehicle for vehicle in vehicles if vehicle!=ego_vehicle]
    try:
        sut = sut[-1]
        acceleration, steering = sut.action['acceleration'],sut.action['steering']
    except:
        acceleration, steering = 0, 0
    return acceleration, steering

def get_wp(env):
    vehicles = env.unwrapped.road.vehicles
    ego_vehicle = env.unwrapped.vehicle
    sut = [vehicle for vehicle in vehicles if vehicle!=ego_vehicle]
    if len(sut) == 0:
        return None
    waypoints = []
    try:
        lanes_index = sut[-1].route
        for lane_index in lanes_index:
            lane = env.unwrapped.road.network.get_lane(lane_index)
            for pos in range(0,int(lane.length)+1,1):
                waypoint = lane.position(pos,0)
                waypoints.append(waypoint)
    except:
        waypoints = get_waypoints_counterpart(env,sut[-1],200)

    state_ = {'route':[],'target_speed':[]}
    state_['route'] = np.array(waypoints)
    state_['target_speed'] = sut[-1].target_speed
    return state_

def get_waypoints_counterpart(env,vehicle,distance):
    current_position = vehicle.position
    # 获取车辆所在车道
    lane_index = env.unwrapped.road.network.get_closest_lane_index(current_position)
    # 获取当前车道对象
    current_lane = env.unwrapped.road.network.get_lane(lane_index)
    
    # 获取前方道路上的waypoints
    waypoints = []
    total_distance = 0
    waypoints.append(current_position)
    while total_distance < distance and current_lane is not None:
        next_position = current_lane.position(current_position[0], 0)  # 假设当前车辆位于车道中心
        waypoints.append(next_position)
        total_distance += np.linalg.norm(next_position - current_position)
        current_position = next_position
    return waypoints

def forbid_rear(state,action):
    #阻止车辆倒车
    vx, vy = state[2], state[3]
    v = np.sqrt(vx**2+vy**2)
    if v <= 0.01 and action<0:
        action = 0
    return action

def state_select(state,env):
    # 筛选出实际的主车sut
    vehicles = env.unwrapped.road.vehicles
    npc_controlled = env.unwrapped.vehicle
    sut = [vehicle for vehicle in vehicles if vehicle!=npc_controlled and vehicle.target_speed>0.5]
    traffic_vehicle = [vehicle for vehicle in vehicles if vehicle!=npc_controlled and vehicle.target_speed<0.5]

    if np.linalg.norm(sut[0].position - npc_controlled.position) < np.linalg.norm(traffic_vehicle[0].position - npc_controlled.position):
        sut_indx = 1 #因为在next
    else:
        sut_indx = 2
        
    state_ = state[[0,sut_indx],:]
    return state_
