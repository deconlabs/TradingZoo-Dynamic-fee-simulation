#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import trading_env
from utils import collect_trajectories, device, clipped_surrogate
from PPOTradingAgent.model import CNNTradingAgent
from common.multiprocessing_env import  SubprocVecEnv


# In[3]:


df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df.fillna(method='ffill', inplace=True)


# In[4]:


# Hyperparameters
class args:
    def __init__(self,no_short):
        self.no_short = no_short
args = args(True)
device = device
learning_rate = 0.001
discount = 0.995
eps = 0.05
K_epoch = 3
num_steps = 128
beta = 0.4
num_envs = 16


# In[5]:


def make_env():
    def _thunk():
        env = trading_env.make(custom_args= args, env_id='training_v1', obs_data_len=256, step_len=16,
                               df=df, fee=0.0, max_position=5, deal_col_name='Price',
                               feature_names=['Price', 'Volume',
                                              'Ask_price', 'Bid_price',
                                              'Ask_deal_vol', 'Bid_deal_vol',
                                              'Bid/Ask_deal', 'Updown'])

        return env

    return _thunk


# In[6]:



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
save_interval = 100

envs = [make_env() for _ in range(num_envs)]
envs = SubprocVecEnv(envs)
model = CNNTradingAgent().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print_interval = 10

scores_list = []
loss_list = []
for n_epi in range(10000):  # 게임 1만판 진행
    n_epi +=1
    loss = 0.0
    log_probs, states, actions, rewards, next_state, masks, values = collect_trajectories(envs,model,num_steps)

    # raise Exception("True" if torch.any(torch.isnan(torch.stack(states))) else "False")
    if beta>0.01:
        beta*=discount
    for _ in range(K_epoch):
        L = -clipped_surrogate(envs,model, log_probs, states, actions, rewards, discount, eps, beta)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        loss+=L.item()
        del L


    score = np.asarray(rewards).sum(axis=0).mean()
    scores_list.append(score)
    loss_list.append(loss)

    if n_epi % print_interval == 0 and n_epi != 0:
        print("# of episode :{}, avg score : {:.4f}, loss : {:.6f}".format(
            n_epi, score / print_interval, loss / print_interval))
        print("actions : ", torch.cat(actions))
        

    if n_epi % save_interval ==0:
        torch.save(model.state_dict(), f'TradingGym_{n_epi}.pth')
        torch.save(scores_list, f"plot/{n_epi}_scores.pth")
        plt.plot(scores_list)
        plt.title("Reward")
        plt.grid(True)
        plt.savefig(f'plot/{n_epi}_ppo.png')
        plt.close()

envs.close()


# In[ ]:




