import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from custom_trading_env import TradingEnv
from utils import device
import DQNTradingAgent.dqn_agent as dqn_agent
from custom_hyperparameters import hyperparams
from arguments import argparser

args = argparser()
device = device
dqn_agent.set_device(device)

save_location = './custom_saves'

# Hyperparameters
# num_steps = 128

save_interval  = 100
print_interval = 5

sample_len   = 4096
obs_data_len = 256
step_len     = 16

n_action_intervals = 10

init_budget = 1

# _persist_period = (obs_data_len // step_len) + int(obs_data_len % step_len != 0)

torch.save(hyperparams, os.path.join(save_location, "hyperparams.pth"))

df = pd.read_hdf('dataset/binance_data.h5', 'STW')
df.fillna(method='ffill', inplace=True)

def main():

    env = TradingEnv(custom_args=args, env_id='custom_trading_env', obs_data_len=obs_data_len, step_len=step_len, sample_len=sample_len,
                           df=df, fee=0.001, initial_budget=1, n_action_intervals=n_action_intervals, deal_col_name='c',
                           feature_names=['o', 'h','l','c','v',
                                          'num_trades', 'taker_base_vol'])
    agent = dqn_agent.Agent(action_size=2 * n_action_intervals + 1, obs_len=obs_data_len, num_features=env.reset().shape[-1], **hyperparams)

    beta = 0.4
    beta_inc = (1 - beta) / 1000
    agent.beta = beta

    scores_list = []
    loss_list = []
    n_epi = 0
    # for n_epi in range(10000):  # 게임 1만판 진행
    while True:
        n_epi +=1

        state = env.reset()
        score = 0.
        actions = []
        rewards = []
#         _reward_deque = deque(maxlen=_persist_period)

        # for t in range(num_steps):
        while True:
            action = int(agent.act(state, eps=0.))
            actions.append(action)
            next_state, reward, done, _ = env.step(action)

#             _reward_deque.append(0)
#             reward = reward - sum(_reward_deque)
#             _reward_deque[-1] = reward

            rewards.append(reward)
            score += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        else:
            agent.memory.reset_multisteps()

        beta = min(1, beta + beta_inc)
        agent.beta = beta

        scores_list.append(score)

        if n_epi % print_interval == 0 and n_epi != 0:
            print_str = "# of episode: {:d}, avg score: {:.4f}\n  Actions: {}".format(n_epi, sum(scores_list[-print_interval:]) / print_interval, np.array(actions))
            print(print_str)
            with open(os.path.join(save_location, "output_log.txt"), mode='a') as f:
                f.write(print_str + '\n')

        if n_epi % save_interval == 0:
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_location, 'TradingGym_Rainbow_{:d}.pth'.format(n_epi)))
            torch.save(scores_list, os.path.join(save_location, 'scores.pth'))

    env.close()


if __name__ == '__main__':
    main()
