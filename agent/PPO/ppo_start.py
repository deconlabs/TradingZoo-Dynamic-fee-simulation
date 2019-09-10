import os
from os.path import dirname
import sys
sys.path.append(dirname(dirname(sys.path[0])))
import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from envs.trading_env_integrated import TradingEnv
from utils import collect_trajectories, clipped_surrogate
from PPOTradingAgent.model import CNNTradingAgent
from common.multiprocessing_env import  SubprocVecEnv
from arguments import argparser

# Hyperparameters

args = argparser()
device = "cuda:" + str(args.device_num) if torch.cuda.is_available() else "cpu"
save_interval = 1000
num_envs = 16
n_episodes   = args.n_episodes
sample_len   = 480
obs_data_len = 256
step_len     = 16
risk_aversion_multiplier = 0.5 + args.risk_aversion / 2
n_action_intervals = 5
init_budget = 1

df = pd.read_hdf('../../dataset/binance_data_train.h5', 'STW')
df.fillna(method='ffill', inplace=True)

save_location = 'saves/Original/{}'.format(args.save_num)
if not os.path.exists(save_location):
    os.makedirs(save_location)

def make_env():
    def _thunk():
        env = TradingEnv(custom_args=args, env_id='custom_trading_env', obs_data_len=obs_data_len, step_len=step_len, sample_len=sample_len,
                           df=df, fee=args.fee, initial_budget=1, n_action_intervals=n_action_intervals, deal_col_name='c', sell_at_end=True,
                           feature_names=['o', 'h','l','c','v',
                                          'num_trades', 'taker_base_vol'])

        return env

    return _thunk

def main():
    learning_rate = 0.001
    discount = 0.995
    beta = 0.4
    eps = 0.05
    K_epoch = 3
    num_steps = 128

    envs = [make_env() for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)
    model = CNNTradingAgent(num_features=envs.reset().shape[-1], n_actions=2 * n_action_intervals + 1).to(device)
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
            torch.save(model.state_dict(), os.path.join(save_location,f'TradingGym_{n_epi}.pth'))
            torch.save(scores_list, os.path.join(save_location,f"{n_epi}_scores.pth"))
            # plt.plot(scores_list)
            # plt.title("Reward")
            # plt.grid(True)
            # plt.savefig(os.path.join(save_location,f'{n_epi}_ppo.png'))
            # plt.close()

    del envs

if __name__ == '__main__':
    main()



