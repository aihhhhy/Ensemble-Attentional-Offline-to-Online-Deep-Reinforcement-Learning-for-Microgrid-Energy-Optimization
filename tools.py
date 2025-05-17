import torch 
import pandas as pd 
import numpy.random as rd
import os 
import numpy as np
from pyomo.core.base.config import default_pyomo_config
from pyomo.core.base.piecewise import Bound
from pyomo.environ import *
from pyomo.opt import SolverFactory
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
from torch.nn.parallel import DataParallel
import torch.nn as nn
import copy

def optimization_base_result(env,month,day,initial_soc,state_record,action_record,next_state_record,reward_record,mask_record,error,cost_list):
    pv=env.data_manager.get_series_pv_data(month,day)
    price=env.data_manager.get_series_price_data(month,day)
    load=env.data_manager.get_series_electricity_cons_data(month,day)
    period=env.episode_length
# parameters
    DG_parameters=env.dg_parameters
    def get_dg_info(parameters):
        p_max=[]
        p_min=[]
        ramping_up=[]
        ramping_down=[]
        a_para=[]
        b_para=[]
        c_para=[]
        
        for name, gen_info in parameters.items():
            p_max.append(gen_info['power_output_max'])
            p_min.append(gen_info['power_output_min'])
            ramping_up.append(gen_info['ramping_up'])
            ramping_down.append(gen_info['ramping_down'])
            a_para.append(gen_info['a'])
            b_para.append(gen_info['b'])
            c_para.append(gen_info['c'])
        return p_max,p_min,ramping_up,ramping_down,a_para,b_para,c_para
    p_max,p_min,ramping_up,ramping_down,a_para,b_para,c_para=get_dg_info(parameters=DG_parameters)
    battery_parameters=env.battery_parameters
    NUM_GEN=len(DG_parameters.keys())
    # NUM_GEN=4
    battery_capacity=env.battery.capacity
    battery_efficiency=env.battery.efficiency

    m=gp.Model("UC")

    #set variables in the system
    on_off=m.addVars(NUM_GEN,period,vtype=GRB.BINARY,name='on_off')
    gen_output=m.addVars(NUM_GEN,period,vtype=GRB.CONTINUOUS,name='output')
    battery_energy_change=m.addVars(period,vtype=GRB.CONTINUOUS,lb=-env.battery.max_charge,ub=env.battery.max_charge,name='battery_action')#directly set constrains for charge/discharge
    grid_energy_import=m.addVars(period,vtype=GRB.CONTINUOUS,lb=0,ub=env.grid.exchange_ability,name='import')# set constrains for exchange between external grid and distributed energy system
    grid_energy_export=m.addVars(period,vtype=GRB.CONTINUOUS,lb=0,ub=env.grid.exchange_ability,name='export')
    soc=m.addVars(period,vtype=GRB.CONTINUOUS,lb=0.2,ub=0.8,name='SOC')

    m.update()
    #1. add balance constrain
    m.addConstrs(((sum(gen_output[g,t] for g in range(NUM_GEN))+pv[t]+grid_energy_import[t]>=load[t]+battery_energy_change[t]+grid_energy_export[t]) for t in range(period)),name='powerbalance')
    #2. add constrain for p max pmin
    m.addConstrs((gen_output[g,t]<=on_off[g,t]*p_max[g] for g in range(NUM_GEN) for t in range(period)),'output_max')
    m.addConstrs((gen_output[g,t]>=on_off[g,t]*p_min[g]for g in range(NUM_GEN) for t in range(period)),'output_min')
    #3. add constrain for ramping up ramping down
    m.addConstrs((gen_output[g,t+1]-gen_output[g,t]<=ramping_up[g] for g in range(NUM_GEN) for t in range(period-1)),'ramping_up')
    m.addConstrs((gen_output[g,t]-gen_output[g,t+1]<=ramping_down[g] for g in range(NUM_GEN) for t in range(period-1)),'ramping_down')
    #4. add constrains for SOC
    m.addConstr(battery_capacity*soc[0]==battery_capacity*initial_soc+(battery_energy_change[0]*battery_efficiency),name='soc0')
    m.addConstrs((battery_capacity*soc[t]==battery_capacity*soc[t-1]+(battery_energy_change[t]*battery_efficiency)for t in range(1,period)),name='soc update')

    # set cost function
    #1 cost of generator
    cost_gen=gp.quicksum((a_para[g]*gen_output[g,t]*gen_output[g,t]+b_para[g]*gen_output[g,t]+c_para[g]*on_off[g,t])for t in range(period) for g in range(NUM_GEN))
    cost_grid_import=gp.quicksum(grid_energy_import[t]*price[t] for t in range(period))
    cost_grid_export=gp.quicksum(grid_energy_export[t]*price[t]*env.sell_coefficient for t in range(period))

    m.setObjective((cost_gen+cost_grid_import-cost_grid_export),GRB.MINIMIZE)
    m.optimize()


    # output_record={'pv':[],'price':[],'load':[],'netload':[],'soc':[],'battery_energy_change':[],'grid_import':[],'grid_export':[],'gen1':[],'gen2':[],'gen3':[],'gen4':[],'step_cost':[]}
    output_record={'pv':[],'price':[],'load':[],'netload':[],'soc':[],'battery_energy_change':[],'grid_import':[],'grid_export':[],'gen1':[],'gen2':[],'gen3':[],'step_cost':[]}
    # new_gen_output = np.array(gen_output)
    new_gen_output = {}

    # 遍历gen_output数组，将属性x的值存储到新数组中
    for row in gen_output:
        new_gen_output[row]=gen_output[row].x


    # 将新的二维数组打印出来
    # for row in new_array:
    #     print(row)
    
    # error = 0
    for i in range(period):
        if new_gen_output[0,i] > 0 and new_gen_output[0,i] < p_min[0]:
            new_gen_output[0,i] = p_min[0]
            # error += 1
        elif new_gen_output[1,i] > 0 and new_gen_output[1,i] < p_min[1]:
            new_gen_output[1,i] = p_min[1]
            # error += 1
        elif new_gen_output[2,i] > 0 and new_gen_output[2,i] < p_min[2]:
            new_gen_output[2,i] = p_min[2]
            # error += 1
        # elif new_gen_output[3,i] > 0 and new_gen_output[3,i] < p_min[3]:
        #     new_gen_output[3,i] = p_min[3]
    
    
    for t in range(period):
        finish = False
        gen_cost=sum((on_off[g,t].x*(a_para[g]*gen_output[g,t].x*gen_output[g,t].x+b_para[g]*gen_output[g,t].x+c_para[g])) for g in range(NUM_GEN))
        grid_import_cost=grid_energy_import[t].x*price[t]
        grid_export_cost=grid_energy_export[t].x*price[t]*env.sell_coefficient
        output_record['pv'].append(pv[t])
        output_record['price'].append(price[t])
        output_record['load'].append(load[t])
        output_record['netload'].append(load[t]-pv[t])
        output_record['soc'].append(soc[t].x)
        output_record['battery_energy_change'].append(battery_energy_change[t].x)
        output_record['grid_import'].append(grid_energy_import[t].x)
        output_record['grid_export'].append(grid_energy_export[t].x)
        output_record['gen1'].append(gen_output[0,t].x)
        output_record['gen2'].append(gen_output[1,t].x)
        output_record['gen3'].append(gen_output[2,t].x)
        # output_record['gen4'].append(gen_output[3,t].x)
        output_record['step_cost'].append((gen_cost+grid_import_cost-grid_export_cost)/1e3)
        reward = 0

        if t==0:
            action_gen1 = np.float32((new_gen_output[0,t])/ramping_up[0])
            action_gen2 = np.float32((new_gen_output[1,t])/ramping_up[1])
            action_gen3 = np.float32((new_gen_output[2,t])/ramping_up[2])
            # action_gen4 = np.float32((new_gen_output[3,t])/ramping_up[3])
            action_battery = np.float32((battery_energy_change[t].x )/ env.battery.max_charge)
            # obs = np.concatenate((np.float32(t),np.float32(price[t]),np.float32(initial_soc),np.float32(load[t]-pv[t]),np.float32(0),np.float32(0),np.float32(0),np.float32(0)),axis=None)
            obs = np.concatenate((np.float32(t),np.float32(price[t]),np.float32(initial_soc),np.float32(load[t]-pv[t]),np.float32(0),np.float32(0),np.float32(0)),axis=None)
            
            # action_battery = np.float32((battery_energy_change[t].x * battery_efficiency)/ env.battery.max_charge)

        else :
            action_gen1 = np.float32((new_gen_output[0,t] - new_gen_output[0,t-1])/ramping_up[0])
            action_gen2 = np.float32((new_gen_output[1,t] - new_gen_output[1,t-1])/ramping_up[1])
            action_gen3 = np.float32((new_gen_output[2,t] - new_gen_output[2,t-1])/ramping_up[2])
            # action_gen4 = np.float32((new_gen_output[3,t] - new_gen_output[3,t-1])/ramping_up[3])
            action_battery = np.float32((battery_energy_change[t].x )/ env.battery.max_charge)
            # obs = np.concatenate((np.float32(t),np.float32(price[t]),np.float32(soc[t-1].x),np.float32(load[t]-pv[t]),np.float32(gen_output[0,t-1].x),np.float32(gen_output[1,t-1].x),np.float32(gen_output[2,t-1].x),np.float32(gen_output[3,t-1].x)),axis=None)
            obs = np.concatenate((np.float32(t),np.float32(price[t]),np.float32(soc[t-1].x),np.float32(load[t]-pv[t]),np.float32(gen_output[0,t-1].x),np.float32(gen_output[1,t-1].x),np.float32(gen_output[2,t-1].x)),axis=None)
            
            
            # action_gen1 = np.float32((new_gen_output[0,t+1] - new_gen_output[0,t])/ramping_up[0])
            # action_gen2 = np.float32((new_gen_output[1,t+1] - new_gen_output[1,t])/ramping_up[1])
            # action_gen3 = np.float32((new_gen_output[2,t+1] - new_gen_output[2,t])/ramping_up[2])
            # action_gen4 = np.float32((new_gen_output[3,t+1] - new_gen_output[3,t])/ramping_up[3])
            # action_battery = np.float32((battery_energy_change[t].x * battery_efficiency)/ env.battery.max_charge)
        if (action_gen1 >1 or action_gen1 < -1):
            error+=1
        elif (action_gen2 >1 or action_gen2 < -1):
            error+=1
        elif (action_gen3 >1 or action_gen3 < -1):
            error+=1
        # elif (action_gen4 >1 or action_gen4 < -1):
        #     error+=1
        if (action_battery >1 or action_battery < -1):
            error+=1
            
        
        state_record.append(obs)
        
        # action=np.concatenate((action_battery, action_gen1, action_gen2, action_gen3, action_gen4),axis=None)
        action=np.concatenate((action_battery, action_gen1, action_gen2, action_gen3),axis=None)
        
        action_record.append(action)
        
        reward-=((gen_cost+grid_import_cost-grid_export_cost)/1e3) 
        reward_record.append(reward)
        
        if t==period-1:
            finish = True
            env.reset()
            month=env.month
            day=env.day
            initial_soc=env.battery.current_capacity
            load=env.data_manager.get_electricity_cons_data(month,day,0) 
            pv_generation=env.data_manager.get_pv_data(month,day,0)
            price=env.data_manager.get_price_data(month,day,0)
            # next_obs=np.concatenate((np.float32(t+1),np.float32(price),initial_soc,np.float32(load-pv_generation),np.float32(0),np.float32(0),np.float32(0),np.float32(0)),axis=None)
            next_obs=np.concatenate((np.float32(t+1),np.float32(price),initial_soc,np.float32(load-pv_generation),np.float32(0),np.float32(0),np.float32(0)),axis=None)
            
        else:
            # next_obs=np.concatenate((np.float32(t+1),np.float32(price[t+1]),np.float32(soc[t].x),np.float32(load[t+1]-pv[t+1]),np.float32(gen_output[0,t].x),np.float32(gen_output[1,t].x),np.float32(gen_output[2,t].x),np.float32(gen_output[3,t].x)),axis=None)
            next_obs=np.concatenate((np.float32(t+1),np.float32(price[t+1]),np.float32(soc[t].x),np.float32(load[t+1]-pv[t+1]),np.float32(gen_output[0,t].x),np.float32(gen_output[1,t].x),np.float32(gen_output[2,t].x)),axis=None)
            
        next_state_record.append(next_obs)
        
        
        mask = (1.0 - int(finish)) * 0.995
        mask_record.append(mask)
            

        
    cost_list.append(sum(output_record['step_cost']))

    state_record_df = pd.DataFrame.from_dict(state_record)
    action_record_df = pd.DataFrame.from_dict(action_record)
    output_record_df = pd.DataFrame.from_dict(output_record)
    # print(action_record)
    return output_record_df,state_record,action_record,next_state_record,reward_record,mask_record,error,cost_list
class Arguments:
    '''revise here for our own purpose'''
    def __init__(self, agent=None, env=None):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training
        self.plot_shadow_on=False# control do we need to plot all shadow figures
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        # self.replace_train_data=True
        self.visible_gpu = '0,1,2,3'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.num_episode = 500
        self.gamma = 0.995  # discount factor of future rewards
        # self.reward_scale = 1  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        # self.learning_rate =1e-6
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        # 软更新的主要特点是每次更新时只更新一小部分目标网络参数，而不是完全替换为新的参数值。
        # 这样可以使目标网络的参数缓慢地向新的参数值靠近，而不是立即完全替换，从而更加平滑地更新网络参数，减少更新的不稳定性。
        self.attention_dim = 16
        self.gru_hidden_dim = 128
        self.net_dim = 256  # the network width 256
        self.batch_size = 4096  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 5  # repeatedly update network to keep critic's loss small 2 ** 5
        self.target_step = 4096 # collect target_step experiences , then update network, 1024
        self.max_memo = 500000  # capacity of replay buffer
        self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.
        self.dropout_rate = 0.2
        # 在强化学习算法中，PER（Prioritized Experience Replay）是一种用于处理稀疏奖励的离线策略评估方法。
        # 这种方法通过优先级体验回放的方式，可以提高对重要经验的采样频率，从而加速学习过程并提高性能。

        '''Arguments for evaluate'''
        # self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        # self.eval_times = 2  # number of times that get episode return in first
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        # self.random_seed_list=[1234,2234,3234,4234,5234]
        self.random_seed_list=[6234]
        '''Arguments for save and plot issues'''
        self.train=True
        self.save_network=True
        self.test_network=True
        self.save_test_data=True
        self.compare_with_pyomo=True
        self.plot_on=True 

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            # self.cwd = f'./{agent_name}'
            self.cwd = f'/kaggle/working/{agent_name}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)# control how many GPU is used 　
def test_one_episode(env, act, device,gail):
    '''to get evaluate information, here record the unblance of after taking action'''
    record_state=[]
    record_action=[]
    record_reward=[]
    record_output=[]
    record_cost=[]
    record_unbalance=[]
    record_system_info=[]# [time price, netload,action,real action, output*4,soc,unbalance(exchange+penalty)]

    record_init_info=[]#should include month,day,time,intial soc,initial
    env.TRAIN = False
    state=env.reset()
    record_init_info.append([env.month,env.day,env.current_time,env.battery.current_capacity])
    print(f'current testing month is {env.month}, day is {env.day},initial_soc is {env.battery.current_capacity}' )
    for i in range(24):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)  
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        real_action=action
        state,next_state,reward, done,cost, = env.step(action,gail,if_gail=False)
        
        record_system_info.append([state[0],state[1],state[3],action,real_action,env.battery.SOC(),env.battery.energy_change,next_state[4],next_state[5],next_state[6],next_state[7],env.unbalance,env.operation_cost])
        # print(action)
        record_state.append(state)
        record_action.append(real_action)
        record_reward.append(reward)
        record_output.append(env.current_output)
        record_unbalance.append(env.unbalance)
        state=next_state
    record_system_info[-1][7:11]=[env.final_step_outputs[0],env.final_step_outputs[1],env.final_step_outputs[2],env.final_step_outputs[3]]
    ## add information of last step soc
    record_system_info[-1][5]=env.final_step_outputs[4]
    record={'init_info':record_init_info,'information':record_system_info,'state':record_state,'action':record_action,'reward':record_reward,'cost':record_cost,'unbalance':record_unbalance,'record_output':record_output}
    return record
def get_con_episode_return(env, agent, act, gail, if_gail, if_att,if_end):
    if if_end:
        episode_return = -500.0  # sum of rewards in an episode
        print("if_end",if_end)
        while episode_return < -40:
            episode_return = 0.0  # sum of rewards in an episode
            episode_unbalance = 0.0
            episode_cost = 0.0
            state = env.reset() 
            for i in range(len(agent.env_list)):  
                    agent.env_list[i] = env.copy(agent.env_list[i])
            states = []
            for i in range(24):
                actions = []
                sub_reward = []
                s_tensor = torch.as_tensor((state,), device=agent.device)
                best_reward = -500
                for j,policy in enumerate(agent.policy_networks):
                    # if i == 0:
                    actions.append(policy(s_tensor)[0].detach().cpu().numpy())
                    # else:
                    #     actions.append(policy(torch.as_tensor((policy_state,), device=agent.device))[0].detach().cpu().numpy())

                    policy_state, policy_next_state, policy_reward, policy_done, policy_cost = agent.env_list[j].step(
                        actions[j], gail, if_gail=False)
                    # total_reward[j] += policy_reward
                    if if_gail == False:
                        agent.total_reward[j] += policy_reward       
                    sub_reward.append(policy_reward)
                    # policy_state = policy_next_state
                    if policy_reward > best_reward :  
                        best_reward = policy_reward
                        best_cost = policy_cost
                        best_index = j
            
                # # print('best_index',best_index)
                actions = np.array(actions)
                # action = (actions[0]+actions[1]+actions[2])/3
                # # weighted_action = np.sum(agent.weights[:, None] * actions, axis=0)
                if best_reward > -1 :
                    state, next_state, reward, done, cost = env.step(actions[best_index], gail, if_gail=False)
                else :
                    best_action = actions[best_index]
                    for n in range(50):
                        agent.env0 = env.copy(agent.env0)
                        # 添加噪声
                        a_tensor = (actions[best_index] + np.random.randn(*actions[best_index].shape) * agent.explore_noise).clip(-1, 1)
                        state, next_state, reward, done, cost = agent.env0.step(a_tensor, gail, if_gail=False)
                        if reward > best_reward :  
                            best_reward = reward
                            best_cost = cost
                            best_action = a_tensor
                    state, next_state, reward, done, cost = env.step(best_action, gail, if_gail=False)
                # state, next_state, reward, done, cost = env.step(actions[best_index], gail, if_gail=False)
                # action = agent.policy_networks[1](s_tensor)[0].detach().cpu().numpy()
                # state, next_state, reward, done, cost = env.step(action, gail, if_gail=False)
                # sub_reward.append(best_reward)
                # states.append(state)
                state = next_state
                for k in range(len(agent.env_list)):  
                    agent.env_list[k] = env.copy(agent.env_list[k])
                episode_return += best_reward
                # episode_return1 += reward
                # print('sub_reward',sub_reward)
                episode_cost += best_cost
                episode_unbalance += env.real_unbalance
                if done:
                    break
    else:
    # if if_end:
    #     print("if_end",if_end)
    
        episode_return = 0.0  # sum of rewards in an episode
        episode_unbalance = 0.0
        episode_cost = 0.0
        state = env.reset() 
        for i in range(len(agent.env_list)):  
                agent.env_list[i] = env.copy(agent.env_list[i])
        states = []
        for i in range(24):
            actions = []
            sub_reward = []
            s_tensor = torch.as_tensor((state,), device=agent.device)
            best_reward = -500
            for j,policy in enumerate(agent.policy_networks):
                # if i == 0:
                actions.append(policy(s_tensor)[0].detach().cpu().numpy())
                # else:
                #     actions.append(policy(torch.as_tensor((policy_state,), device=agent.device))[0].detach().cpu().numpy())

                policy_state, policy_next_state, policy_reward, policy_done, policy_cost = agent.env_list[j].step(
                    actions[j], gail, if_gail=False)
                # total_reward[j] += policy_reward
                if if_gail == False:
                    agent.total_reward[j] += policy_reward       
                sub_reward.append(policy_reward)
                # policy_state = policy_next_state
                if policy_reward > best_reward :  
                    best_reward = policy_reward
                    best_cost = policy_cost
                    best_index = j
            # # print('best_index',best_index)
            actions = np.array(actions)
            # action = (actions[0]+actions[1]+actions[2])/3
            # # weighted_action = np.sum(agent.weights[:, None] * actions, axis=0)
            if best_reward > -1 :
                state, next_state, reward, done, cost = env.step(actions[best_index], gail, if_gail=False)
            else :
                best_action = actions[best_index]
                for n in range(50):
                    agent.env0 = env.copy(agent.env0)
                    # 添加噪声
                    a_tensor = (actions[best_index] + np.random.randn(*actions[best_index].shape) * agent.explore_noise).clip(-1, 1)
                    state, next_state, reward, done, cost = agent.env0.step(a_tensor, gail, if_gail=False)
                    if reward > best_reward :  
                        best_reward = reward
                        best_cost = cost
                        best_action = a_tensor
                state, next_state, reward, done, cost = env.step(best_action, gail, if_gail=False)
            # state, next_state, reward, done, cost = env.step(actions[best_index], gail, if_gail=False)
            # action = agent.policy_networks[1](s_tensor)[0].detach().cpu().numpy()
            # state, next_state, reward, done, cost = env.step(action, gail, if_gail=False)
            # sub_reward.append(best_reward)
            # states.append(state)
            state = next_state
            for k in range(len(agent.env_list)):  
                agent.env_list[k] = env.copy(agent.env_list[k])
            episode_return += best_reward
            # episode_return1 += reward
            # print('sub_reward',sub_reward)
            episode_cost += best_cost
            episode_unbalance += env.real_unbalance
            if done:
                break
    print("performance_record",agent.performance_record)
    agent.update_performance_record(agent.total_reward)
    # 更新完成一个episode后，添加到累计奖励列表
    agent.cumulative_rewards.append(episode_return)
    return episode_return, episode_unbalance, episode_cost
# def get_con_episode_return(env,agent, act, device,gail,if_gail,if_att,total_reward):
#     episode_return = 0.0  # sum of rewards in an episode
#     episode_unbalance=0.0
#     episode_cost = 0.0
#     state = env.reset()
    
#     # total_reward = np.zeros(agent.num_networks)

#     for i in range(24):
#         actions = []
#         s_tensor = torch.as_tensor((state,), device=device)

#         for j, policy in enumerate(agent.policy_networks):
            
#             if i == 0:
#                 actions.append(policy(s_tensor)[0].detach().cpu().numpy())
#             else :
#                 actions.append(policy(torch.as_tensor((policy_state,), device=device))[0].detach().cpu().numpy())     
#             policy_state, policy_next_state, policy_reward, policy_done,policy_cost,= env.step(policy(s_tensor)[0].detach().cpu().numpy(),gail ,if_gail=False)
#             total_reward[j] += policy_reward
#             policy_state = policy_next_state
#         actions = np.array(actions)
#         weighted_action = np.sum(agent.weights[:, None] * actions, axis=0)     
#         # 添加噪声
#         a_tensor = (weighted_action + np.random.randn(*weighted_action.shape) * agent.explore_noise).clip(-1, 1)
        
#         # a_tensor = act(s_tensor)           
#         # action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
#         state, next_state, reward, done,cost,= env.step(a_tensor,gail ,if_gail=False)
#         state=next_state
#         episode_return += reward
        
#         episode_cost += cost
#         episode_unbalance+=env.real_unbalance
#         if done:
#             break
#     print("total_reward:",total_reward)
#     agent.update_performance_record(total_reward)
#     return episode_return,episode_unbalance,episode_cost

# def get_episode_return(env,agent, act, device,gail,if_gail,if_att,total_reward):
#     episode_return = 0.0  # sum of rewards in an episode
#     episode_unbalance=0.0
#     episode_cost = 0.0
#     state = env.reset()
#     for i in range(24):
#         s_tensor = torch.as_tensor((state,), device=device)
#         a_tensor = act(s_tensor)
#         action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
#         state, next_state, reward, done,cost,= env.step(action,gail ,if_gail=False)
#         state=next_state
#         episode_return += reward
#         episode_cost += cost
#         episode_unbalance+=env.real_unbalance
#         if done:
#             break
#     return episode_return,episode_unbalance,episode_cost


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state    # 将一条轨迹的状态加到状态缓冲区
            self.buf_other[self.next_idx:next_idx] = other    # 将一条轨迹的其它信息加到其它缓冲区
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size) 
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])
    # def sample_batch(self, batch_size) -> tuple:
    #     start_index = np.random.randint(0, self.now_len - batch_size + 1)
    #     order_indices = np.arange(start_index, start_index + batch_size)
    #     r_m_a = self.buf_other[order_indices]
    #     return (r_m_a[:, 0:1],
    #             r_m_a[:, 1:2],
    #             r_m_a[:, 2:],
    #             self.buf_state[order_indices],
    #             self.buf_state[order_indices + 1])
    def order_sample_batch(self, batch_size) -> tuple:
        order_indices = np.arange(0, batch_size)
        r_m_a = self.buf_other[order_indices]    
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[order_indices],
                self.buf_state[order_indices+1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx
        
    def reset_buffer(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')
