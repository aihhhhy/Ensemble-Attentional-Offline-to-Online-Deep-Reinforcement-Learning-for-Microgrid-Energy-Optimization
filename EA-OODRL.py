import os
import pickle
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
import random
from torch.nn.modules import loss
from random_generator_battery import ESSEnv
import pandas as pd 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Parameters import battery_parameters,dg_parameters
import seaborn as sns
from tools import Arguments,test_one_episode,ReplayBuffer,optimization_base_result,get_con_episode_return
from agent import AgentSAC
from random_generator_battery import ESSEnv
from net import *
from copy import deepcopy
from torch.nn.parallel import DataParallel




def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
    ary_other = torch.as_tensor([item[1] for item in _trajectory])
    ary_other[:, 0] = ary_other[:, 0]   # ten_reward
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
    return _steps, _r_exp

    
def save_state(agent, buffer, episode, return_record, loss_record, path,env):
    agent.save_model(path,episode)
    random_state = random.getstate()
    numpy_random_state = rd.get_state()
    torch_random_state = torch.get_rng_state()
    with open(os.path.join(path, f'random_state_{episode}.pkl'), 'wb') as f:
        pickle.dump((random_state, numpy_random_state,torch_random_state), f)
    agent.save_replay_buffer(path, buffer,episode)
    with open(os.path.join(path, f'return_record_{episode}.pkl'), 'wb') as f:
        pickle.dump(return_record, f)
    with open(os.path.join(path, f'loss_record_{episode}.pkl'), 'wb') as f:
        pickle.dump(loss_record, f)
    env_state = env.save_internal_state()
    with open(os.path.join(path, f'env_state_{episode}.pkl'), 'wb') as f:
        pickle.dump(env_state, f)
    print(f"State saved to {path} at episode {episode}")
    # with open(os.path.join(path, f'env_state_{episode}.pkl'), 'wb') as f:
    #     pickle.dump(env, f)

def load_state(agent, episode, path, env):
    agent.load_model(path,episode)
    with open(os.path.join(path, f'random_state_{episode}.pkl'), 'rb') as f:
        random_state, numpy_random_state,torch_random_state = pickle.load(f)
    random.setstate(random_state)
    rd.set_state(numpy_random_state)
    torch.set_rng_state(torch_random_state)
 
    with open(os.path.join(path, f'env_state_{episode}.pkl'), 'rb') as f:
        env_state = pickle.load(f)
    env.load_internal_state(env_state)
    # with open(os.path.join(path, f'env_state_{episode}.pkl'), 'rb') as f:
    #     env = pickle.load(f)

    buffer = agent.load_replay_buffer(path,episode)
    with open(os.path.join(path, f'return_record_{episode}.pkl'), 'rb') as f:
        return_record = pickle.load(f)
    with open(os.path.join(path, f'loss_record_{episode}.pkl'), 'rb') as f:
        loss_record = pickle.load(f)
    print(f"State loaded from {path} at episode {episode}")
    return  buffer, return_record, loss_record



class Discriminator(nn.Module):
    def __init__(self, state_dim, mid_dim, action_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
    def forward(self, x, a):
        return torch.sigmoid(self.net(torch.cat((x, a), dim=1)))
    
    
class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim)
        self.discriminator = nn.DataParallel(self.discriminator)
        self.discriminator = self.discriminator.to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.module.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, mask, device,rewards,i_episode,buffer,batch_size):       
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        expert_actions = torch.tensor(expert_a).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        agent_actions = torch.tensor(agent_a).to(device)
        
        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        # print("Device IDs: ", self.discriminator.device_ids)
        
        discriminator_loss = nn.BCELoss()(
        agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
            expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
       
        transition_dict = {
        'states': agent_states,
        'actions': agent_actions,
        'rewards': rewards,
        'next_states': next_s,
        'mask': mask
        }
       
        critic_loss,actor_loss,entropy_loss = self.agent.update_net_gall(args.repeat_times, args.soft_update_tau, transition_dict,buffer,batch_size)
        return critic_loss,actor_loss,entropy_loss



if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args=Arguments()
    return_record={'episode':[],'steps':[],'mean_episode_reward':[],'cost':[],'unbalance':[]}
    loss_record={'episode':[],'steps':[],'critic_loss':[],'actor_loss':[],'entropy_loss':[]}
    expert_record={'states':[], 'action':[],'next_states':[],'rewards':[],'mask':[]}   
    
    args.visible_gpu='0'    
    gpu_id = 0
    args.agent=AgentSAC()
    agent=args.agent
    # device = torch.device(f"cuda:{args.visible_gpu}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    agent.device = device
    if_gail = True
    if_att = True
    beta = 5.0
    num_random = 32
    # load_path = '/kaggle/input/energy-systems/con-gpus-batch-DRL-for-Energy-Systems-Optimal-Scheduling-main/non_muti'
    # save_path = '/kaggle/working/saved_states'
    load_path = 'non_muti'
    save_path = 'non_muti'
    os.makedirs(save_path, exist_ok=True)
    
    
    for seed in args.random_seed_list:
        args.random_seed = seed
        agent_name=f'{args.agent.__class__.__name__}'
        args.agent.cri_target=True
        args.env=ESSEnv()
        args.init_before_training(if_main=True)
        '''init agent and environment'''
        env=args.env
        agent.init(args.net_dim,env.state_space.shape[0],env.action_space.shape[0],args.attention_dim,args.dropout_rate,beta,num_random,args.learning_rate,args.if_per_or_gae==True)
        '''init replay buffer'''
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                              action_dim= env.action_space.shape[0])
        '''start training'''
        cwd=args.cwd
        gamma=args.gamma
        batch_size=args.batch_size# how much data should be used to update net
        target_step=args.target_step#how manysteps of one episode should stop
        repeat_times=args.repeat_times# how many times should update for one batch size data
        soft_update_tau = args.soft_update_tau
        agent.state=env.reset()
        '''collect data and train and update network'''
        num_episode=args.num_episode
        '''here record real unbalance'''
        seq_length = 16
        # 计算 num_sequences
        num_sequences = batch_size // seq_length

        # 采集专家数据
        expert_episode = 1000
        states = []
        actions = []  
        expert_states = []
        expert_actions = [] 
        next_state_record = []
        reward_record = []
        mask_record = []
        # cost_list = []
        # error = 0

        with open(r'Expert_data/expert_data.pkl', "rb") as f:
            object = pickle.load(f, encoding='latin1')
        # with open(r'/kaggle/input/energy-systems/con-gpus-batch-DRL-for-Energy-Systems-Optimal-Scheduling-main/Expert_data/expert_data_g3_part1.pkl', "rb") as f:
        #     object = pickle.load(f, encoding='latin1')
        expert_states = object['states']
        expert_actions = object['action']
        next_state_record = object['next_states']
        reward_record = object['rewards']
        mask_record = object['mask']
        expert_done = [1.0 - i / 0.995 for i in mask_record]
      

        sorted_indices = np.argsort(-np.array(reward_record))


        top_20_len = int(len(sorted_indices) * 0.6)
        # top_indices = sorted_indices[:4096*3]
        top_indices = sorted_indices[:top_20_len]
        

        top_states = np.array([expert_states[i] for i in top_indices])
        top_actions = np.array([expert_actions[i] for i in top_indices])
        top_next_states = np.array([next_state_record[i] for i in top_indices])
        top_rewards = np.array([reward_record[i] for i in top_indices])
        top_masks = np.array([mask_record[i] for i in top_indices])
        top_done = np.array([expert_done[i] for i in top_indices])
        
        expert_s = torch.as_tensor(top_states, dtype=torch.float32)
        expert_a = torch.as_tensor(top_actions, dtype=torch.float32)
        expert_next_s = torch.as_tensor(top_next_states, dtype=torch.float32)
        expert_r = torch.as_tensor(top_rewards, dtype=torch.float32)
        expert_mask = torch.as_tensor(top_masks, dtype=torch.float32)
        expert_done = torch.as_tensor(top_done, dtype=torch.float32)     
        print(expert_r) 
        # GAIL
        gail = GAIL(agent,env.state_space.shape[0],env.action_space.shape[0], args.net_dim, args.learning_rate, agent.device)
        n_episode = 100
        return_list = []
        rate = 0
        e_len = 0
        e_lens = 0
        agent.total_reward = np.zeros(agent.num_networks)
        agent.state=env.reset()
        if args.train:    
    
            saved_files = [f for f in os.listdir(load_path) if f.startswith('random_state_')]
            if saved_files:
                latest_episode = max([int(f.split('_')[2].split('.')[0]) for f in saved_files])
                start_episode = latest_episode
                buffer, return_record, loss_record = load_state(agent, start_episode, load_path,env)
            else:
                start_episode = 0
            for i in range(start_episode, n_episode, 100):
                for i_episode in range(i, i + 100):
                    agent.state = env.reset()           
                    if i_episode >= n_episode:
                        break
                    collect_data=True        
                    if i_episode < 50:
                        while collect_data:
                            print(f'buffer:{buffer.now_len}')                    
                            with torch.no_grad():                          
                                trajectory=agent.explore_env(env,target_step,gail, if_gail,if_att)
                                steps,r_exp=update_buffer(trajectory)
                                buffer.update_now_len()                                                                           
                            if buffer.now_len>=4096:
                                collect_data=False
                        reward, mask, action, state, next_s = buffer.order_sample_batch(batch_size)                 
                        critic_loss,actor_loss,entropy_loss = gail.learn(expert_s, expert_a, state, action, next_s, mask, agent.device,reward, i_episode,buffer,batch_size)                   
                        buffer.reset_buffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                                action_dim= env.action_space.shape[0])
                        buffer.update_now_len()
                    else:   
                        if_att = False
                        if_gail=False
                        # agent.learning_rate = 2 ** -16
                        collect_data=True 
                        while collect_data:                       
                            print(f'buffer:{buffer.now_len}')
                            with torch.no_grad():
                                trajectory=agent.explore_env(env,target_step,gail, if_gail,if_att)
                                steps,r_exp=update_buffer(trajectory)
                                buffer.update_now_len()
                            if buffer.now_len>=10000:
                                collect_data=False 
                    
                        reward, mask, action, state, next_s = buffer.sample_batch(batch_size)          
                        transition_dict = {
                        'states': state,
                        'actions': action,
                        'rewards': reward,
                        'next_states': next_s,
                        'mask': mask}
                        critic_loss,actor_loss,entropy_loss=agent.update_net(buffer,batch_size,repeat_times,soft_update_tau)
                        if i_episode % 10==0:   #每十轮再将新策略跑出的路径放缓冲区
                            with torch.no_grad():
                                trajectory=agent.explore_env(env,target_step,gail,if_gail,if_att)
                                steps,r_exp=update_buffer(trajectory)

                    loss_record['critic_loss'].append(critic_loss)
                    loss_record['actor_loss'].append(actor_loss)
                    loss_record['entropy_loss'].append(entropy_loss)
                    with torch.no_grad():
                        episode_reward,episode_unbalance,episode_cost=get_con_episode_return(env,agent,agent.act,gail,if_gail,if_att)
                        return_record['mean_episode_reward'].append(episode_reward)
                        return_record['unbalance'].append(episode_unbalance)
                        return_record['cost'].append(episode_cost)
                    print(f'curren epsiode is {i_episode}, reward:{episode_reward}, cost:{episode_cost}, unbalance:{episode_unbalance},buffer_length: {buffer.now_len}')
                    if (i_episode + 1) % 100 == 0:
                        save_state(agent, buffer, i_episode + 1, return_record, loss_record, save_path,env)
                
                
     