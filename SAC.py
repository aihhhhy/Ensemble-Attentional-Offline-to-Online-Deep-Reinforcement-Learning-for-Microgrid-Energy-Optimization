import os
import pickle
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
from random_generator_battery import ESSEnv
import pandas as pd 

from tools import Arguments,get_episode_return,test_one_episode,ReplayBuffer,optimization_base_result
from agent import AgentSAC
from random_generator_battery import ESSEnv

def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
    ary_other = torch.as_tensor([item[1] for item in _trajectory])
    ary_other[:, 0] = ary_other[:, 0]   # ten_reward
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
    return _steps, _r_exp

if __name__=='__main__':
    args=Arguments()
    reward_record={'episode':[],'steps':[],'mean_episode_reward':[],'cost':[],'unbalance':[]}
    loss_record={'episode':[],'steps':[],'critic_loss':[],'actor_loss':[],'entropy_loss':[]}
    args.visible_gpu='0'
    if_gail = False
    for seed in args.random_seed_list:
        args.random_seed = seed
        args.agent=AgentSAC()
        agent_name=f'{args.agent.__class__.__name__}'
        args.agent.cri_target=True
        args.env=ESSEnv()
        args.init_before_training(if_main=True)
        '''init agent and environment'''
        agent=args.agent
        env=args.env
        agent.init(args.net_dim,env.state_space.shape[0],env.action_space.shape[0],args.learning_rate,args.if_per_or_gae)
        '''init replay buffer'''
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                              action_dim= env.action_space.shape[0])
        '''start training'''
        cwd=args.cwd
        gamma=args.gamma
        batch_size=args.batch_size# how much data should be used to update net
        target_step=args.target_step#how manysteps of one episode should stop 4096

        repeat_times=args.repeat_times# how many times should update for one batch size data

        soft_update_tau = args.soft_update_tau

        agent.state=env.reset()

        '''collect data and train and update network'''
        num_episode=args.num_episode
        '''here record real unbalance'''

        ##
        # args.train=False
        # args.save_network=False
        # args.test_network=False
        # args.save_test_data=False
        # args.compare_with_pyomo=False
        #
        if args.train:
            collect_data=True
            while collect_data:
                print(f'buffer:{buffer.now_len}')
                with torch.no_grad():
                    trajectory=agent.explore_env(env,target_step,gail= None,if_gail = False)
                    steps,r_exp=update_buffer(trajectory)
                    buffer.update_now_len()
                if buffer.now_len>=10000:
                    collect_data=False
            for i_episode in range(num_episode):
                critic_loss,actor_loss,entropy_loss=agent.update_net(buffer,batch_size,repeat_times,soft_update_tau)  #训练网络，返回cri损失，策略损失，熵正则项系数
                loss_record['critic_loss'].append(critic_loss)
                loss_record['actor_loss'].append(actor_loss)
                loss_record['entropy_loss'].append(entropy_loss)
                with torch.no_grad():
                    episode_reward,episode_unbalance,episode_cost=get_episode_return(env,agent.act,agent.device,gail= None,if_gail = False)
                    reward_record['mean_episode_reward'].append(episode_reward)
                    reward_record['unbalance'].append(episode_unbalance)
                    reward_record['cost'].append(episode_cost)
                print(f'curren epsiode is {i_episode}, reward:{episode_reward}, cost:{episode_cost}, unbalance:{episode_unbalance},buffer_length: {buffer.now_len}')
                if i_episode % 10==0:   #每十轮再将新策略跑出的路径放缓冲区
                # target_step
                    with torch.no_grad():
                        trajectory=agent.explore_env(env,target_step,gail= None,if_gail = False)
                        steps,r_exp=update_buffer(trajectory)
    act_save_path = f'{args.cwd}/actor.pth'
    loss_record_path=f'{args.cwd}/loss_data.pkl'
    reward_record_path=f'{args.cwd}/reward_data.pkl'
    with open (loss_record_path,'wb') as tf:
        pickle.dump(loss_record,tf)
    with open (reward_record_path,'wb') as tf:
        pickle.dump(reward_record,tf)



    if args.save_network:
        torch.save(agent.act.state_dict(),act_save_path)
        print('actor parameters have been saved')
    
    if args.test_network:
        args.cwd=agent_name
        agent.act.load_state_dict(torch.load(act_save_path))
        print('parameters have been reload and test')
        record=test_one_episode(env,agent.act,agent.device,gail = None)
        eval_data=pd.DataFrame(record['information'])
        eval_data.columns=['time_step','price','netload','action','real_action','soc','battery','gen1','gen2','gen3','unbalance','operation_cost']
    if args.save_test_data:
        test_data_save_path=f'{args.cwd}/test_data.pkl'
        with open(test_data_save_path,'wb') as tf:
            pickle.dump(record,tf)


    # with open(r'E:\pyProject\Energy-System\DRL-for-Energy-Systems-Optimal-Scheduling-main\DRL-for-Energy-Systems-Optimal-Scheduling-main\AgentSAC\test_data.pkl', "rb") as f:
    #     object = pickle.load(f, encoding='latin1')
    # df = pd.DataFrame(pd.DataFrame.from_dict(object, orient='index').values.T, columns=list(object.keys()))
    # df.to_csv(r'E:\document\Matlab R2022a\test_data_SAC.csv')
    
    # with open(r'E:\pyProject\Energy-System\DRL-for-Energy-Systems-Optimal-Scheduling-main\DRL-for-Energy-Systems-Optimal-Scheduling-main\AgentSAC\loss_data.pkl', "rb") as f:
    #     object = pickle.load(f, encoding='latin1')
    # df = pd.DataFrame(pd.DataFrame.from_dict(object, orient='index').values.T, columns=list(object.keys()))
    # df.to_csv(r'E:\document\Matlab R2022a\loss_data_SAC.csv')
    
    # with open(r'E:\pyProject\Energy-System\DRL-for-Energy-Systems-Optimal-Scheduling-main\DRL-for-Energy-Systems-Optimal-Scheduling-main\AgentSAC\reward_data.pkl', "rb") as f:
    #     object = pickle.load(f, encoding='latin1')
    # df = pd.DataFrame(pd.DataFrame.from_dict(object, orient='index').values.T, columns=list(object.keys()))
    # df.to_csv(r'E:\document\Matlab R2022a\reward_data_SAC.csv')
    with open(r'/kaggle/working/AgentSAC/test_data.pkl', "rb") as f:
        object = pickle.load(f, encoding='latin1')
    df = pd.DataFrame(pd.DataFrame.from_dict(object, orient='index').values.T, columns=list(object.keys()))
    df.to_csv(r'/kaggle/working/AgentSAC/test_data_TD3.csv')
    
    with open(r'/kaggle/working/AgentSAC/loss_data.pkl', "rb") as f:
        object = pickle.load(f, encoding='latin1')
    df = pd.DataFrame(pd.DataFrame.from_dict(object, orient='index').values.T, columns=list(object.keys()))
    df.to_csv(r'/kaggle/working/AgentSAC/loss_data_TD3.csv')
    
    with open(r'/kaggle/working/AgentSAC/reward_data.pkl', "rb") as f:
        object = pickle.load(f, encoding='latin1')
    df = pd.DataFrame(pd.DataFrame.from_dict(object, orient='index').values.T, columns=list(object.keys()))
    df.to_csv(r'/kaggle/working/AgentSAC/reward_data_TD3.csv')



    '''compare with pyomo data and results'''
    if args.compare_with_pyomo:
        month=record['init_info'][0][0]
        day=record['init_info'][0][1]
        initial_soc=record['init_info'][0][3]   
        print(initial_soc)     
        base_result,state_record,action_record=optimization_base_result(env,month,day,initial_soc,state_record=[],action_record=[])
    if args.plot_on:
        from plotDRL import PlotArgs,make_dir,plot_evaluation_information,plot_optimization_result
        plot_args=PlotArgs()
        plot_args.feature_change='2000Episode_100exchange_50penalty'
        args.cwd=agent_name
        plot_dir=make_dir(args.cwd,plot_args.feature_change)
        plot_optimization_result(base_result,plot_dir)
        plot_evaluation_information(args.cwd+'/'+'test_data.pkl',plot_dir)
    '''compare the different cost get from pyomo and SAC'''
    ration=sum(eval_data['operation_cost'])/sum(base_result['step_cost'])
    print(sum(eval_data['operation_cost']))
    print(sum(base_result['step_cost']))
    print(ration)   