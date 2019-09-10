import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.trading_env_integrated import TradingEnv
import DQNTradingAgent.dqn_agent as dqn_agent
from custom_hyperparameters import hyperparams
from arguments import argparser

args = argparser()
# device_num, save_num, risk_aversion, n_episodes, fee

device = torch.device("cuda:{}".format(args.device_num))
dqn_agent.set_device(device)

load_location = 'saves/Original/{}'.format(args.save_num)

if args.environment=="default":
    save_location = 'saves/transfer/{}/{}'.format(args.fee, args.save_num)
else:
    save_location = 'saves/transfer/{}/{}'.format(args.environment , args.save_num)

if not os.path.exists(save_location):
    os.makedirs(save_location)

save_interval  = 100
print_interval = 1

n_episodes   = args.n_episodes
sample_len   = 480
obs_data_len = 192
step_len     = 1
fee          = args.fee
sell_at_end  = False

risk_aversion_multiplier = 0.5 + args.risk_aversion / 2

n_action_intervals = 5

init_budget = 1

torch.save(hyperparams, os.path.join(save_location, "hyperparams.pth"))
torch.save({'n_episodes': n_episodes, 'sample_len': sample_len, 'obs_data_len': obs_data_len, 'step_len': step_len,
            'fee': fee, 'risk_aversion_multiplier': risk_aversion_multiplier ,'n_action_intervals': n_action_intervals,
            'init_budget': init_budget},
           os.path.join(save_location, "config.pth"))

df = pd.read_hdf('dataset/binance_data_train.h5', 'STW')
df.fillna(method='ffill', inplace=True)

def main():

    env = TradingEnv(custom_args=args, env_id='custom_trading_env', obs_data_len=obs_data_len, step_len=step_len, sample_len=sample_len,
                     df=df, fee=fee, initial_budget=1, n_action_intervals=n_action_intervals, deal_col_name='c',
                     sell_at_end=sell_at_end,
                     feature_names=['o', 'h','l','c','v',
                                    'num_trades', 'taker_base_vol'])
    agent = dqn_agent.Agent(action_size=2 * n_action_intervals + 1, obs_len=obs_data_len, num_features=env.reset().shape[-1], **hyperparams)
    agent.qnetwork_local.load_state_dict(torch.load(os.path.join(load_location, 'TradingGym_Rainbow_1000.pth'), map_location=device))
    agent.qnetwork_local.to(device)

    beta = 0.4
    beta_inc = (1 - beta) / 1000
    agent.beta = beta

    scores_list = []
    
    
    for i_episode in range(1,n_episodes+1):
        state = env.reset()
        score = 0.
        actions = []
        rewards = []

        # for t in range(num_steps):
        while True:
            action = int(agent.act(state, eps=0.))
            next_state, reward, done, _ , fee_rate = env.step(action)

            rewards.append(reward)
            score += reward
            if reward < 0:
                reward *= risk_aversion_multiplier
            if done:
                action = 2 * n_action_intervals
            actions.append(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        else:
            agent.memory.reset_multisteps()

        beta = min(1, beta + beta_inc)
        agent.beta = beta

        scores_list.append(score)

        if i_episode % print_interval == 0 and i_episode != 0:
            print_str = "# of episode: {:d}, avg score: {:.4f}\n  Actions: {}".format(i_episode, sum(scores_list[-print_interval:]) / print_interval, np.array(actions))
            print(print_str)
            # with open(os.path.join(save_location, "output_log.txt"), mode='a') as f:
            #     f.write(print_str + '\n')

        if i_episode % save_interval == 0:
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_location, 'TradingGym_Rainbow_{:d}.pth'.format(i_episode)))
            torch.save(scores_list, os.path.join(save_location, 'scores.pth'))

    del env


if __name__ == '__main__':
    main()
