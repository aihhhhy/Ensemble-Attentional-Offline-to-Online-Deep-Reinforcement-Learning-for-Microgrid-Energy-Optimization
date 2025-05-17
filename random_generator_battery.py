
import random
import numpy as np
import pandas as pd 
import gym
from gym import spaces 
import torch 
from net import *
from tools import Arguments
from Parameters import battery_parameters,dg_parameters
import copy

class Constant:
	MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	MAX_STEP_HOURS = 24 * 30
class DataManager():
    def __init__(self) -> None:
        # 数据集中的历史数据数组
        self.PV_Generation=[]
        self.Prices=[]
        self.Electricity_Consumption=[]
    def add_pv_element(self,element):self.PV_Generation.append(element)
    def add_price_element(self,element):self.Prices.append(element)
    def add_electricity_element(self,element):self.Electricity_Consumption.append(element)

    # get current time data based on given month day, and day_time   随机取出历史数据中某一小时的记录
    def get_pv_data(self,month,day,day_time):return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_price_data(self,month,day,day_time):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_electricity_cons_data(self,month,day,day_time):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]  # 取出数据中具体某个时刻的用电量=（随机某月份的随机某天的某一小时）
    # get series data for one episode
    def get_series_pv_data(self,month,day): return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_price_data(self,month,day):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_electricity_cons_data(self,month,day):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]

class DG():
    '''simulate a simple diesel generator here'''
    def __init__(self,parameters):
        # self.name=parameters.keys()
        self.name = list(parameters.keys())[0]
        self.a_factor=parameters['a']
        self.b_factor=parameters['b']
        self.c_factor=parameters['c']
        self.power_output_max=parameters['power_output_max']
        self.power_output_min=parameters['power_output_min']
        self.ramping_up=parameters['ramping_up']
        self.ramping_down=parameters['ramping_down']
        self.last_step_output=None 
    def step(self,action_gen):
        
        # ramping_up功率增加或减少的能力
        output_change=action_gen*self.ramping_up#   输出功率变化量，根据 action_gen 的正负值和大小，可以决定输出功率是增加还是减少，以及变化的幅度。如果发生负荷突然增加的情况，需要发电单元快速增加输出功率以满足需求
        output=self.current_output+output_change   # 新全部输电功率
        if output>0:
            output=max(self.power_output_min,min(self.power_output_max,output))# meet the constrain 
        else:
            output=0
        self.current_output=output
    def _get_cost(self,output):
        if output<=0:
            cost=0
        else:
            cost=(self.a_factor*pow(output,2)+self.b_factor*output+self.c_factor)
        return cost 
    def reset(self):
        self.current_output=0   # 当前输出功率
    def save_state(self):
        return {
            'name': self.name,
            'a_factor': self.a_factor,
            'b_factor': self.b_factor,
            'c_factor': self.c_factor,
            'power_output_max': self.power_output_max,
            'power_output_min': self.power_output_min,
            'ramping_up': self.ramping_up,
            'ramping_down': self.ramping_down,
            'current_output': self.current_output
        }

    def load_state(self, state):
        self.name = state['name']
        self.a_factor = state['a_factor']
        self.b_factor = state['b_factor']
        self.c_factor = state['c_factor']
        self.power_output_max = state['power_output_max']
        self.power_output_min = state['power_output_min']
        self.ramping_up = state['ramping_up']
        self.ramping_down = state['ramping_down']
        self.current_output = state['current_output']

class Battery():
    '''simulate a simple battery here'''
    def __init__(self,parameters):
        self.capacity=parameters['capacity']
        self.max_soc=parameters['max_soc']
        self.initial_capacity=parameters['initial_capacity']
        self.min_soc=parameters['min_soc']# 0.2
        self.degradation=parameters['degradation']# degradation cost 1.2 退化
        self.max_charge=parameters['max_charge']# nax charge ability
        self.max_discharge=parameters['max_discharge']
        self.efficiency=parameters['efficiency']
    def step(self,action_battery):
        energy=action_battery*self.max_charge     #充/放电量  action_battery的±代表充放电
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))     #当前电量+充放电量/最大容量，并作了最大/小电量比率限制
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge, positive, if discharge, negative
        self.current_capacity=updated_capacity# update capacity to current codition  电量占比
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        cost=energy**2*self.degradation
        return cost  
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity=np.random.uniform(0.2,0.8) #生成一个介于0.2和0.8之间的随机数 初始化电池容量
    def save_state(self):
        return {
            'capacity': self.capacity,
            'max_soc': self.max_soc,
            'initial_capacity': self.initial_capacity,
            'min_soc': self.min_soc,
            'degradation': self.degradation,
            'max_charge': self.max_charge,
            'max_discharge': self.max_discharge,
            'efficiency': self.efficiency,
            'current_capacity': self.current_capacity
        }

    def load_state(self, state):
        self.capacity = state['capacity']
        self.max_soc = state['max_soc']
        self.initial_capacity = state['initial_capacity']
        self.min_soc = state['min_soc']
        self.degradation = state['degradation']
        self.max_charge = state['max_charge']
        self.max_discharge = state['max_discharge']
        self.efficiency = state['efficiency']
        self.current_capacity = state['current_capacity']
class Grid():
    def __init__(self):
        
        self.on=True
        if self.on:
            self.exchange_ability=100
        else:
            self.exchange_ability=0
    def _get_cost(self,current_price,energy_exchange):
        return current_price*energy_exchange
    def retrive_past_price(self):
        result=[]
        if self.day<1:
            past_price=self.past_price#
        else:
            past_price=self.price[24*(self.day-1):24*self.day]
            # print(past_price)
        for item in past_price[(self.time-24)::]:
            result.append(item)
        for item in self.price[24*self.day:(24*self.day+self.time)]:
            result.append(item)
        return result 
class ESSEnv(gym.Env):
    def __init__(self,**kwargs):            #接受任意数量的关键字参数。关键字参数会被收集到一个名为kwargs的字典中
        super(ESSEnv,self).__init__()
        #parameters 
        self.data_manager=DataManager()
        self._load_year_data()    # 到这历史数据加载进来了并赋值给 self.PV_Generation=[], self.Prices=[], self.Electricity_Consumption=[]此三个数组
        self.episode_length=kwargs.get('episode_length',24)  
        self.month=None
        self.day=None
        self.TRAIN=True
        self.current_time=None
        self.battery_parameters=kwargs.get('battery_parameters',battery_parameters)
        self.dg_parameters=kwargs.get('dg_parameters',dg_parameters)
        self.penalty_coefficient=50#control soft penalty constrain 
        self.sell_coefficient=0.5# control sell benefits

        self.grid=Grid()
        self.battery=Battery(self.battery_parameters)
        self.dg1=DG(self.dg_parameters['gen_1'])
        self.dg2=DG(self.dg_parameters['gen_2'])
        self.dg3=DG(self.dg_parameters['gen_3'])
        # self.dg4=DG(self.dg_parameters['gen_4'])

        # self.action_space=spaces.Box(low=-1,high=1,shape=(5,),dtype=np.float32)  #Box(-1.0, 1.0, (4,), float32)
        self.action_space=spaces.Box(low=-1,high=1,shape=(4,),dtype=np.float32)  #Box(-1.0, 1.0, (4,), float32)

        self.state_space=spaces.Box(low=0,high=1,shape=(7,),dtype=np.float32) #spaces.Box是一个OpenAI Gym库中定义的用于表示连续空间的类。
        # self.state_space=spaces.Box(low=0,high=1,shape=(8,),dtype=np.float32) #spaces.Box是一个OpenAI Gym库中定义的用于表示连续空间的类。

    @property
    def netload(self):

        return self.demand-self.grid.wp_gen-self.grid.pv_gen
        
    def reset(self,):
        self.month=np.random.randint(1,13)# here we choose 12 month
        if self.TRAIN:
            self.day=np.random.randint(1,20)
        else:
            self.day=np.random.randint(20,Constant.MONTHS_LEN[self.month-1])
        self.current_time=0
        self.battery.reset()
        self.dg1.reset()
        self.dg2.reset()
        self.dg3.reset()
        # self.dg4.reset()
        return self._build_state()
    def _build_state(self):
        soc=self.battery.SOC()
        dg1_output=self.dg1.current_output
        dg2_output=self.dg2.current_output
        dg3_output=self.dg3.current_output
        # dg4_output=self.dg4.current_output
        time_step=self.current_time
        electricity_demand=self.data_manager.get_electricity_cons_data(self.month,self.day,self.current_time) 
        pv_generation=self.data_manager.get_pv_data(self.month,self.day,self.current_time)
        price=self.data_manager.get_price_data(self.month,self.day,self.current_time)
        net_load=electricity_demand-pv_generation
        obs=np.concatenate((np.float32(time_step),np.float32(price),np.float32(soc),np.float32(net_load),np.float32(dg1_output),np.float32(dg2_output),np.float32(dg3_output)),axis=None)
        # obs=np.concatenate((np.float32(time_step),np.float32(price),np.float32(soc),np.float32(net_load),np.float32(dg1_output),np.float32(dg2_output),np.float32(dg3_output),np.float32(dg4_output)),axis=None)
        return obs

    def step(self,action,gail,if_gail):# state transition here current_obs--take_action--get reward-- get_finish--next_obs
        ## here we want to put take action into each components
        # from GAIL import Discriminator
        args = Arguments()
        gpu_id = 0
        args.visible_gpu='0'
        device = torch.device(f"cuda:{args.visible_gpu}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        # self.discriminator = Discriminator(self.state_space.shape[0], args.net_dim,self.action_space.shape[0]).to(device)
        current_obs=self._build_state()
        self.battery.step(action[0])# here execute the state-transition part, battery.current_capacity also changed
        self.dg1.step(action[1])
        self.dg2.step(action[2])
        self.dg3.step(action[3])
        # self.dg3.step(action[4])
        # truely corresonding to the result 
        # (-self.battery.energy_change)加负值的原因：若电池放电，则self.battery.energy_change是负值，这里计算输出多少电量，所以添-
        
        current_output=np.array((self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,-self.battery.energy_change))
        # current_output=np.array((self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,self.dg4.current_output,-self.battery.energy_change))
        
        self.current_output=current_output
        actual_production=sum(current_output)        
        netload=current_obs[3]  #net_load
        price=current_obs[1]

        unbalance=actual_production-netload

        reward=0
        excess_penalty=0
        deficient_penalty=0
        sell_benefit=0
        buy_cost=0
        self.excess=0
        self.shedding=0
        if unbalance>=0:# it is now in excess condition
            if unbalance<=self.grid.exchange_ability:   #富余能量可与电网交换
                sell_benefit=self.grid._get_cost(price,unbalance)*self.sell_coefficient #sell money to grid is little [0.029,0.1]
            else:
                sell_benefit=self.grid._get_cost(price,self.grid.exchange_ability)*self.sell_coefficient
                #real unbalance that even grid could not meet 
                self.excess=unbalance-self.grid.exchange_ability
                excess_penalty=self.excess*self.penalty_coefficient  #能量过剩，超过电网的交换能力，设置过剩惩罚
        else:# unbalance <0, its load shedding model, in this case, deficient penalty is used 
            if abs(unbalance)<=self.grid.exchange_ability:
                buy_cost=self.grid._get_cost(price,abs(unbalance))
            else:
                buy_cost=self.grid._get_cost(price,self.grid.exchange_ability)
                self.shedding=abs(unbalance)-self.grid.exchange_ability  #大于0代表电网可以解决不平衡，小于0代表不能解决不平衡
                deficient_penalty=self.shedding*self.penalty_coefficient
        battery_cost=self.battery._get_cost(self.battery.energy_change)# we set it as 0 this time 
        dg1_cost=self.dg1._get_cost(self.dg1.current_output)
        dg2_cost=self.dg2._get_cost(self.dg2.current_output)
        dg3_cost=self.dg3._get_cost(self.dg3.current_output)
        # dg4_cost=self.dg4._get_cost(self.dg4.current_output)

        #将上面得到的总和除以1000（即将其缩小1000倍）  我们想要最小化成本即最小化reware，加上惩罚项，相当于违背我们的希望，让reware增加,所以为了达到最小化成本的目的，要设法消除供需不平衡，才能更快达到最小化成本的目的
        # reward为正时可以代表收益(电量卖给电网)大于成本 , 为负时收益(电量卖给电网)小于成本
        
        
        if if_gail:
            agent_states = torch.tensor([current_obs], dtype=torch.float).to(device)
            agent_actions = torch.tensor([action], dtype=torch.float).to(device)
            agent_prob = gail.discriminator(agent_states, agent_actions)
            reward -= torch.log(agent_prob).detach().cpu().numpy()
        else:          
            reward-=(battery_cost+dg1_cost+dg2_cost+dg3_cost+excess_penalty+
            deficient_penalty-sell_benefit+buy_cost)/1e3   
            # reward-=(battery_cost+dg1_cost+dg2_cost+dg3_cost +dg4_cost +excess_penalty+
            # deficient_penalty-sell_benefit+buy_cost)/1e3   
        
        
        self.operation_cost=(battery_cost+dg1_cost+dg2_cost+dg3_cost+excess_penalty+
        deficient_penalty-sell_benefit+buy_cost)/1e3
        # self.operation_cost=(battery_cost+dg1_cost+dg2_cost+dg3_cost+ dg4_cost+excess_penalty+
        # deficient_penalty-sell_benefit+buy_cost)/1e3
        self.unbalance=unbalance
        self.real_unbalance=self.shedding+self.excess
        final_step_outputs=[self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,self.battery.current_capacity]
        # final_step_outputs=[self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,self.dg4.current_output,self.battery.current_capacity]
        
        self.current_time+=1
        finish=(self.current_time==self.episode_length) #24
        if finish:
            self.final_step_outputs=final_step_outputs
            self.current_time=0
            # self.day+=1
            # if self.day>Constant.MONTHS_LEN[self.month-1]:
            #     self.day=1
            #     self.month+=1
            # if self.month>12:
            #     self.month=1
            #     self.day=1
            next_obs=self.reset()
            
        else:
            next_obs=self._build_state()
        return current_obs,next_obs,float(reward),finish, self.operation_cost #返回当前状态，下一个状态，奖励，是否结束
    def render(self, current_obs, next_obs, reward, finish,cost):
        print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, cost={},terminal={}\n'.format(self.day,self.current_time, current_obs, next_obs, reward, cost,finish))
    def _load_year_data(self):
        # pv_df=pd.read_csv('data/PV.csv',sep=';')
        # #hourly price data for a year 
        # price_df=pd.read_csv('data/Prices.csv',sep=';')
        # # mins electricity consumption data for a year 全部转为一维数组形式
        # electricity_df=pd.read_csv('data/H4.csv',sep=';')
        
        pv_df=pd.read_csv('/kaggle/input/energy-systems1/con-gpus-batch-DRL-for-Energy-Systems-Optimal-Scheduling-main/data/PV.csv',sep=';')
        #hourly price data for a year 
        price_df=pd.read_csv('/kaggle/input/energy-systems1/con-gpus-batch-DRL-for-Energy-Systems-Optimal-Scheduling-main/data/Prices.csv',sep=';')
        # mins electricity consumption data for a year 全部转为一维数组形式
        electricity_df=pd.read_csv('/kaggle/input/energy-systems1/con-gpus-batch-DRL-for-Energy-Systems-Optimal-Scheduling-main/data/H4.csv',sep=';')
        

        
        pv_data=pv_df['P_PV_'].apply(lambda x: x.replace(',','.')).to_numpy(dtype=float) # 年度光伏发电量
        price=price_df['Price'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        electricity=electricity_df['Power'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        # netload=electricity-pv_data
        '''we carefully redesign the magnitude for price and amount of generation as well as demand'''
        # 对每个小时的光伏发电量数值 * 200   最后显示一小时几十或上百的数值
        for element in pv_data:
            self.data_manager.add_pv_element(element*200)
            
        # 对电价只取十分位，且若十分位数 < 0.5，取0.5 所以数组中只会出现0.5，1，2，3....
        for element in price:
            element/=10
            if element<=0.5:
                element=0.5
            self.data_manager.add_price_element(element)
            
        # 对一个小时(60分钟)的用电需求量加起来 * 300  最后显示一小时几十或上百的数值
        for i in range(0,electricity.shape[0],60):
            element=electricity[i:i+60]
            self.data_manager.add_electricity_element(sum(element)*300)
            
    
    def save_internal_state(self):
        return {
       
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
        }

    def load_internal_state(self, state):
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        
  
    #     return new_instance 
    def copy(self,new_instance):  
        new_instance.month=copy.deepcopy(self.month)
        new_instance.day=copy.deepcopy(self.day)
        new_instance.current_time = copy.deepcopy(self.current_time)
        new_instance.battery = copy.deepcopy(self.battery)  
        new_instance.grid = copy.deepcopy(self.grid)  
        new_instance.data_manager = copy.deepcopy(self.data_manager) 
        new_instance.dg1 = copy.deepcopy(self.dg1) 
        new_instance.dg2 = copy.deepcopy(self.dg2) 
        new_instance.dg3 = copy.deepcopy(self.dg3) 
        
        # 复制其他必要的属性  
        return new_instance 
if __name__ == '__main__':
    env=ESSEnv()
    env.TRAIN=False
    rewards=[]

    current_obs=env.reset()
    tem_action=[0.1,0.1,0.1,0.1,0.1]
    for _ in range (144):
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        current_obs,next_obs,reward,finish,cost=env.step(tem_action)
        env.render(current_obs,next_obs,reward,finish,cost)
        current_obs=next_obs
        rewards.append(reward)
