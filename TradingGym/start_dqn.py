import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import trading_env
from utils import device
import DQNTradingAgent.dqn_agent as dqn_agent
from hyperparameters import hyperparams
from arguments import argparser

args = argparser()
device = device
dqn_agent.set_device(device)


# Hyperparameters
num_steps = 128

save_interval  = 100
print_interval = 5

torch.save(hyperparams, "./DQN_logs_and_saves/hyperparams.pth")

df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df.fillna(method='ffill', inplace=True)

def main():

    env = trading_env.make(custom_args = args , env_id='training_v1', obs_data_len=256, step_len=16,
                           df=df, fee=0.0, max_position=5, deal_col_name='Price',
                           feature_names=['Price', 'Volume',
                                          'Ask_price', 'Bid_price',
                                          'Ask_deal_vol', 'Bid_deal_vol',
                                          'Bid/Ask_deal', 'Updown'])
    agent = dqn_agent.Agent(action_size=3, num_features=16, **hyperparams)

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

        for t in range(num_steps):
            action = int(agent.act(state, eps=0.))
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            score += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        else:
            agent.memory.reset_multisteps()

        scores_list.append(score)

        if n_epi % print_interval == 0 and n_epi != 0:
            print_str = "# of episode: {:d}, avg score: {:.4f}\n  Actions: {}".format(n_epi, sum(scores_list[-print_interval:]) / print_interval, np.array(actions))
            print(print_str)
            with open("./DQN_logs_and_saves/output_log.txt", mode='a') as f:
                f.write(print_str + '\n')

        if n_epi % save_interval == 0:
            torch.save(agent.qnetwork_local.state_dict(), './DQN_logs_and_saves/TradingGym_Rainbow_{:d}.pth'.format(n_epi))
            torch.save(scores_list, './DQN_logs_and_saves/scores.pth')

    envs.close()


if __name__ == '__main__':
    main()
