from net import *
import os 
import numpy.random as rd 
from copy import deepcopy
from tools import Arguments,test_one_episode,ReplayBuffer,optimization_base_result
from torch.nn.parallel import DataParallel
import torch.nn as nn
import pickle
import torch.nn.functional as F
from random_generator_battery import ESSEnv




class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = 0.05
        self.trajectory_list = None
        
        self.num_networks = 3  # 设置集成策略网络的数量
        self.policy_networks = []  # 策略网络列表
        self.policy_target_networks = [] # 目标策略网络列表
        self.cri_networks = []
        self.cri_target_networks = []
        self.performance_record = [0.0] * self.num_networks  # 初始化表现记录
        self.weights = np.ones(self.num_networks) / self.num_networks  # 初始化权重
        self.smooth_rewards = [0.0] * self.num_networks  # 初始化滑动平均奖励
        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct =None
        # self.act_ = self.act_target_ = self.if_use_act_target_ = self.act_optim_ = self.ClassAct_ =None
        
        
    
    def init(self, net_dim, state_dim, action_dim, attention_dim,dropout_rate,learning_rate, _if_per_or_gae=False, gpu_id=0 ):
        # explict call self.init() for multiprocessing
        # self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim)
        self.cri = nn.DataParallel(self.cri)
        self.cri = self.cri.to(self.device)
        print("cri_Device IDs: ", self.cri.device_ids)
        
        # 初始化多个策略网络
        for i in range(self.num_networks):
            # device = torch.device(f"cuda:{gpu_ids[i % len(gpu_ids)]}" if (torch.cuda.is_available() and (gpu_ids[i % len(gpu_ids)] >= 0)) else "cpu")
            
            act = self.ClassAct(net_dim, state_dim, action_dim, attention_dim)
            act = nn.DataParallel(act)   
            act = act.to(self.device)        
            act_target = deepcopy(act) if self.if_use_act_target else act
            self.policy_networks.append(act)
            self.policy_target_networks.append(act_target)
            print("act_Device IDs: ", act.device_ids)
            
            # self.policy_networks.append((act, device))
            # self.policy_target_networks.append((act_target, device))
   
            
        # self.act = self.ClassAct(net_dim, state_dim, action_dim,attention_dim).to(self.device) if self.ClassAct else self.cri
        
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        # self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act
        # self.act_target_ = deepcopy(self.act_) if self.if_use_act_target_ else self.act_
        print(learning_rate)
        self.cri_optim = torch.optim.Adam(self.cri.module.parameters(), self.learning_rate)  #制定优化器规则
        # self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
       
        self.act_optim = [torch.optim.Adam(net.module.parameters(), self.learning_rate) for net in self.policy_networks]
        # self.cri_optim = [torch.optim.Adam(cri_net.parameters(), learning_rate) for cri_net in self.cri_networks]
        # self.act_optim_ = torch.optim.Adam(self.act_.parameters(), learning_rate) if self.ClassAct_ else self.cri
        self.performance_record = [0] * self.num_networks  # 记录每个策略网络的表现
        self.best_performance_index = None
        self.total_reward = None
        # del self.ClassCri, self.ClassAct,self.ClassAct_
        del self.ClassCri, self.ClassAct
        


    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
 
        action = self.act(states)[0]  
    
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1) #加噪声，每个噪声从截断正态分布中抽取

        return action.detach().cpu().numpy()
  

    def explore_env(self, env, target_step, if_gail):
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            
            state,next_state, reward, done, cost= env.step(action,if_gail)
           
            trajectory.append((state, (reward, done, *action)))
            state = env.reset() if done else next_state
        self.state = state
        return trajectory
    
    
    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()   # 计算梯度
        optimizer.step()       # 更新价值网络参数

    @staticmethod
    def soft_update(target_net, current_net, tau):    #打包网络参数进行软更新
        for tar, cur in zip(target_net.module.parameters(), current_net.module.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def save_model(self, save_path,episode):
        model_data = {
            'policy_networks': [net.module.state_dict() for net in self.policy_networks],
            'policy_target_networks': [net.module.state_dict() for net in self.policy_target_networks],
            'cri': self.cri.module.state_dict(),
            'cri_target': self.cri_target.module.state_dict(),
            'cri_optim': self.cri_optim.state_dict(),
            'act_optim': [optim.state_dict() for optim in self.act_optim],
            # 'total_reward': self.total_reward,
            'performance_record': self.performance_record,
            'weights': self.weights
        }
        torch.save(model_data, f"{save_path}/model_{episode}.pth")
        print(f"Model saved to {save_path}/model_{episode}.pth")

    def load_model(self, load_path,episode):
        model_data = torch.load(f"{load_path}/model_{episode}.pth", map_location=self.device)
        for i, net in enumerate(self.policy_networks):
            net.module.load_state_dict(model_data['policy_networks'][i])
        for i, net in enumerate(self.policy_target_networks):
            net.module.load_state_dict(model_data['policy_target_networks'][i])
        self.cri.module.load_state_dict(model_data['cri'])
        self.cri_target.module.load_state_dict(model_data['cri_target'])
        self.cri_optim.load_state_dict(model_data['cri_optim'])
        for i, optim in enumerate(self.act_optim):
            optim.load_state_dict(model_data['act_optim'][i])
        # self.total_reward = model_data['total_reward']
        self.total_reward = model_data['performance_record']
        # self.weights = model_data['weights']
        print(f"Model loaded from {load_path}/model_{episode}.pth")

    def save_replay_buffer(self, save_path, replay_buffer,episode):
        with open(f"{save_path}/replay_buffer_{episode}.pkl", 'wb') as f:
            pickle.dump(replay_buffer, f)
        print(f"Replay buffer saved to {save_path}/replay_buffer_{episode}.pkl")

    def load_replay_buffer(self, load_path,episode):
        with open(f"{load_path}/replay_buffer_{episode}.pkl", 'rb') as f:
            replay_buffer = pickle.load(f)
        print(f"Replay buffer loaded from {load_path}/replay_buffer_{episode}.pkl")
        return replay_buffer
class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticTwin
        # self.ClassAct = gru_ActorSAC
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False
        # self.if_use_act_target_ = False
        
        self.noise_std = 0.1  # σ
        self.noise_clip = 0.2  # c
        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.args = Arguments()
        
        self.noise_std = 0.2
        self.noise_clip = 0.5
        self.reward_target = -1
        self.adjustment_factor = 0.1
        self.reg_weight = 0.1
        self.cumulative_rewards = []
        self.episodes_for_stability = 5
        self.env0 = ESSEnv()  
        env1 = ESSEnv()  
        env2 = ESSEnv()  
        env3 = ESSEnv()
        self.env_list = [env1, env2, env3] 
        

    def init(self, net_dim, state_dim, action_dim,attention_dim,gru_hidden_dim,beta,num_random,learning_rate=1e-6, _if_use_per=False, gpu_id=0, env_num=1):
        # super().init(net_dim, state_dim, action_dim, attention_dim,learning_rate, _if_use_per, gpu_id)
        super().init(net_dim, state_dim, action_dim, attention_dim,gru_hidden_dim,learning_rate, _if_use_per, gpu_id)
        

        #将初始的熵正则化参数α设置为一个合适的初始值,将熵正则化参数作为可训练参数，SAC算法可以自动地学习合适的探索程度
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter action_dim=4
        
        #创建一个Adam优化器来更新self.alpha_log元组
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        
        self.target_entropy = np.log(action_dim)   #设置目标熵
        
        self.beta = beta  # CQL损失函数中的系数
        self.num_random = num_random  # CQL中的动作采样数

  
    def update_weights(self):
        performance_sum = np.sum(self.performance_record)
        if performance_sum == 0:
            self.weights = np.ones(self.num_networks) / self.num_networks
        else:
            self.weights = 1 - (np.array(self.performance_record) / performance_sum)
 
            
    def select_action(self, state,if_att,env,gail,if_gail) -> np.ndarray:
        
        actions = []
     
        best_reward = -500
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)

        # if if_att:        
        for i,policy in enumerate(self.policy_networks):
            actions.append(policy(states)[0].detach().cpu().numpy())
            # self.env_list[i] = env.copy(self.env_list[i])  
            policy_state, policy_next_state, policy_reward, policy_done, policy_cost = self.env_list[i].step(
                actions[i], gail, if_gail)    
            if policy_reward > best_reward :  
                best_reward = policy_reward
                best_index = i
        # action = self.policy_networks[1](states)[0].detach().cpu().numpy()

        actions = np.array(actions)
        return actions[best_index]    
       
 
       
    def update_performance_record(self, total_reward):
        # 根据策略网络的表现更新记录
        # 这里假设策略网络按照顺序执行并获取各自的total_reward
        for i in range(self.num_networks):
            self.performance_record[i] = total_reward[i]
        self.update_weights()
 


    def explore_env(self, env, target_step,gail,if_gail,if_att):
        trajectory = list()
        state = self.state
        for i in range(len(self.env_list)):  
            self.env_list[i] = env.copy(self.env_list[i])  
            
        for _ in range(target_step):
            action = self.select_action(state, if_att,env,gail,if_gail)
            
            state,next_state, reward, done,cost = env.step(action,gail ,if_gail)

            trajectory.append((state, (reward, done, *action)))
            if done :
                state = env.reset()
                for i in range(len(self.env_list)):  
                    self.env_list[i] = env.copy(self.env_list[i])  
            else:
                state = next_state
        # print("1 target_step")
        self.state = state
        return trajectory
    
    def diversity_loss(self):
        diversity_loss = 0
        for i in range(self.num_networks - 1):
            for j in range(i + 1, self.num_networks):
                policy_i = self.policy_networks[i]
                policy_j = self.policy_networks[j]
                dist_i = policy_i(self.state)
                dist_j = policy_j(self.state)
                kl_divergence = F.kl_div(F.log_softmax(dist_i, dim=-1), 
                                         F.softmax(dist_j, dim=-1), 
                                         reduction='batchmean')
                diversity_loss += kl_divergence

        return diversity_loss
   
    
    def add_noise_to_action(self, next_a):
        epsilon = torch.randn_like(next_a) * self.noise_std  
        epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)  
        perturbed_next_a = next_a + epsilon  
        return perturbed_next_a

    def adjust_noise(self, reward):
       
        if reward < self.reward_target:
            self.noise_std = min(1.0, self.noise_std + self.adjustment_factor)
            # print(f"Reward {reward} < Target {self.reward_target}: Increasing noise to {self.noise_std}")
        else:
            self.noise_std = max(0.1, self.noise_std - self.adjustment_factor)
            # print(f"Reward {reward} >= Target {self.reward_target}: Decreasing noise to {self.noise_std}")

    def interact_with_environment(self, state, env):
        next_a = self.get_action(state)  
        perturbed_next_a = self.add_noise_to_action(next_a)
        next_state, reward, done, _ = env.step(perturbed_next_a)  
        self.adjust_noise(reward)  
        return next_state, reward, done
    

    def adjust_reg_weight(self, diversity_metric, stability_metric):
   
        diversity_threshold = 0.1  
        stability_threshold = 0.2  

    
        if diversity_metric < diversity_threshold:
            self.reg_weight *= 1.1  
        elif stability_metric < stability_threshold:
            self.reg_weight *= 0.9  
        
        # 将reg_weight限制在合理范围内
        self.reg_weight = min(max(self.reg_weight, 0.001), 1.0)

    def calculate_diversity(self, state):
      
        actions = []
        for policy in self.policy_networks:
            action = policy(state).detach()
            actions.append(action)

        diversity = 0.0
        num_pairs = 0
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                diversity += torch.norm(actions[i] - actions[j], p=2).mean()
                num_pairs += 1

        # 计算平均距离
        if num_pairs > 0:
            diversity /= num_pairs

        return diversity

    def calculate_stability(self):
        # 如果已经收集到足够多的episode，计算累计奖励的方差并调整reg_weight
        if len(self.cumulative_rewards) >= self.episodes_for_stability:
            stability = torch.var(torch.tensor(self.cumulative_rewards))
            self.cumulative_rewards = []
        else :
            stability = torch.tensor(0.0)  # 如果样本数太少，认为稳定性很好
        return stability
        
    def calculate_regularization(self,state):
        regularization = 0.0
        for i in range(len(self.policy_networks)):
            for j in range(i + 1, len(self.policy_networks)):
                action_i = self.policy_networks[i].module.get_action(state).detach()
                action_j = self.policy_networks[j].module.get_action(state).detach()
                regularization += -torch.norm(action_i - action_j, p=2).mean()  # 负号是为了反向最优化

        # 对正则化项进行标准化
        if len(self.policy_networks) > 1:
            regularization /= (len(self.policy_networks) * (len(self.policy_networks) - 1) / 2)

        return regularization

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()
        alpha = self.alpha_log.exp().detach()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            i = 0
            for net, target_net, optimizer in zip(self.policy_networks, self.policy_target_networks, self.act_optim):
                # i += 1
                # if i == 2:
                with torch.no_grad():
                    next_a, next_log_prob = target_net.module.get_action_logprob(next_s)
                    next_q = torch.min(*self.cri_target.module.get_q1_q2(next_s, next_a))
                    q_label = reward + mask * (next_q + next_log_prob * alpha)
                    
                q1, q2 = self.cri.module.get_q1_q2(state, action)
                obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
                self.optim_update(self.cri_optim, obj_critic)
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

                action_pg, log_prob = net.module.get_action_logprob(state)
                obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
                self.optim_update(self.alpha_optim, obj_alpha)

                alpha = self.alpha_log.exp().detach()
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
                obj_actor = -(torch.min(*self.cri_target.module.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()
                
                # 计算策略多样性和稳定性
                diversity_metric = self.calculate_diversity(state)
                stability_metric = self.calculate_stability()

                # 调整reg_weight
                self.adjust_reg_weight(diversity_metric, stability_metric)
                
                # 添加正则化项
                obj_actor += self.reg_weight * self.calculate_regularization(state)
                
                self.optim_update(optimizer, obj_actor)
                self.soft_update(target_net, net, soft_update_tau)
                # break
                # print(self.reg_weight * self.calculate_regularization(state))
        
        return obj_critic.item(), obj_actor.item(), alpha.item()

    
    
# 指导GAIL更新网络
    def update_net_gall(self, repeat_times, soft_update_tau, transition_dict,buffer,batch_size):
         # 选择一个合适的序列长度，例如32
        seq_length = 16
        # 计算 num_sequences
        num_sequences = batch_size // seq_length
        buffer.update_now_len() 
        alpha = self.alpha_log.exp().detach()   # tensor([0.0231])e的alpha_log次方，1除以4的e次方，detach()张量从计算图中分离出来。这样做是为了保留exp()计算的结果，但不再需要其梯度信息，从而避免对之后的计算产生影响。
        obj_critic = obj_actor = None
        
        for _ in range(int(2**4 )): #训练轮次    
            with torch.no_grad():
                    state = torch.tensor(transition_dict['states'],
                                dtype=torch.float).to(self.device)
                    action = torch.tensor(transition_dict['actions'],dtype=torch.float).to(self.device)
                    reward = torch.tensor(transition_dict['rewards'],
                                        dtype=torch.float).view(-1, 1).to(self.device)
                    next_s = torch.tensor(transition_dict['next_states'],
                                            dtype=torch.float).to(self.device)
                    mask = torch.tensor(transition_dict['mask'],
                                        dtype=torch.float).view(-1, 1).to(self.device)
            # for net, target_net, optimizer,cri,cri_target, cri_optim in zip(self.policy_networks, self.policy_target_networks, self.act_optim,self.cri_networks,self.cri_target_networks,self.cri_optim):
            for net, target_net, optimizer in zip(self.policy_networks, self.policy_target_networks, self.act_optim):
            
                '''objective of critic (loss function of critic)'''
                with torch.no_grad():
                  
                    next_a, next_log_prob = target_net.module.get_action_logprob(next_s)  
                    
                    next_q = torch.min(*self.cri_target.module.get_q1_q2(next_s, next_a))  
                    q_label = reward + mask * (next_q + next_log_prob * alpha)        
                q1, q2 = self.cri.module.get_q1_q2(state, action)    
                obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)   
                self.optim_update(self.cri_optim, obj_critic)     
                self.soft_update(self.cri_target, self.cri, soft_update_tau)  
                '''objective of alpha (temperature parameter automatic adjustment)'''   
                # state = state.reshape(num_sequences, seq_length, 8)
                
                action_pg, log_prob = net.module.get_action_logprob(state)  # policy gradient   
                # action_pg = action_pg.reshape(batch_size, 5)
                # log_prob = log_prob.reshape(batch_size, 1)
                obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()  
                self.optim_update(self.alpha_optim, obj_alpha)    
                
                alpha = self.alpha_log.exp().detach()
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
                obj_actor = -(torch.min(*self.cri_target.module.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()
                diversity_metric = self.calculate_diversity(state)
                stability_metric = self.calculate_stability()

                self.adjust_reg_weight(diversity_metric, stability_metric)
                
              
                obj_actor += self.reg_weight * self.calculate_regularization(state)
                
                self.optim_update(optimizer, obj_actor)
                self.soft_update(target_net, net, soft_update_tau)
       

        return obj_critic.item(), obj_actor.item(), alpha.item()
        

    def save_or_load_agent(self, cwd, if_save, replay_buffer=None):
        if if_save:
            self.save_model(cwd)
            if replay_buffer:
                self.save_replay_buffer(cwd, replay_buffer)
        else:
            self.load_model(cwd)
            if replay_buffer:
                return self.load_replay_buffer(cwd)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

