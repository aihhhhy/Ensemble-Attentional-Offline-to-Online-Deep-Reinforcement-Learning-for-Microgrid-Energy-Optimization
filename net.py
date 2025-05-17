import torch 
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
import torch.distributions as dist



class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0) #限制动作的取值范围，避免动作过大或过小。


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, attention_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),)
        self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()  #截断，防止标准差过大或过小
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize 从给定的正态分布中采样一个动作值，不可导

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)  
        a_std = a_std_log.exp() #标准差

        noise = torch.randn_like(a_avg, requires_grad=True)  #重参数化 从标准正态分布中随机抽样与a_avg相同形状的张量噪音
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh() 高斯分布中采样动作a

        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5   ln(策略Π)
        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()  f=ln(策略Π)+constant修正项
        return a_tan, log_prob.sum(1, keepdim=True)

class gru_ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, attention_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())

        self.attention = nn.MultiheadAttention(mid_dim, num_heads=attention_dim)

        self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))
        self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))

        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        # batch_size, state_dim = state.size()
        tmp = self.net_state(state)
        tmp = tmp.unsqueeze(0)  # Adding sequence length dimension for attention

        attn_out, _ = self.attention(tmp, tmp, tmp)
        attn_out = attn_out.squeeze(0)  # Removing sequence length dimension

        return self.net_a_avg(attn_out).tanh()

    def get_action(self, state):
        # batch_size, state_dim = state.size()
        t_tmp = self.net_state(state)
        t_tmp = t_tmp.unsqueeze(0)  # Adding sequence length dimension for attention

        attn_out, _ = self.attention(t_tmp, t_tmp, t_tmp)
        attn_out = attn_out.squeeze(0)  # Removing sequence length dimension

        a_avg = self.net_a_avg(attn_out)
        a_std = self.net_a_std(attn_out).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()

    def get_action_logprob(self, state):
        # batch_size, state_dim = state.size()
        t_tmp = self.net_state(state)
        t_tmp = t_tmp.unsqueeze(0)  # Adding sequence length dimension for attention

        attn_out, _ = self.attention(t_tmp, t_tmp, t_tmp)
        attn_out = attn_out.squeeze(0)  # Removing sequence length dimension

        a_avg = self.net_a_avg(attn_out)
        a_std_log = self.net_a_std(attn_out).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()

        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2) * 0.5
        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()
        return a_tan, log_prob.sum(1, keepdim=True)
    
    def get_action_distribution(self, state):
        t_tmp = self.net_state(state)
        t_tmp = t_tmp.unsqueeze(0)  # Adding sequence length dimension for attention

        attn_out, _ = self.attention(t_tmp, t_tmp, t_tmp)
        attn_out = attn_out.squeeze(0)  # Removing sequence length dimension

        a_avg = self.net_a_avg(attn_out)
        a_std_log = self.net_a_std(attn_out).clamp(-20, 2)
        a_std = a_std_log.exp()

        # 创建一个由均值和标准差定义的正态分布
        action_distribution = dist.Normal(a_avg, a_std)
        return action_distribution
 



class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state):
        return self.net(state)  # advantage value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
    

class Discriminator(nn.Module):
    def __init__(self, state_dim, mid_dim, action_dim):
        # super(Discriminator, self).__init__()
        super().__init__()
        # self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
    def forward(self, x, a):
        # cat = torch.cat([x, a], dim=1)
        # x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.net(torch.cat((x, a), dim=1)))
    
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = self.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5))
        attention_output = attention_weights @ V
        return attention_output

