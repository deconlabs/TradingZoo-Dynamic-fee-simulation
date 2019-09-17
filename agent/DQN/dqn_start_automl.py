import os
from os.path import dirname
import sys
sys.path.append(dirname(dirname(sys.path[0])))
import importlib
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import logger
import nni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import DQNTradingAgent.dqn_agent as dqn_agent
from envs.trading_env_integrated import TradingEnv
from custom_hyperparameters import hyperparams
from arguments import argparser


args = argparser() # device_num, save_num, risk_aversion, n_episodes

torch.cuda.manual_seed_all(7)

device = torch.device("cuda:{}".format(args.device_num))
dqn_agent.set_device(device)

save_location = 'saves/Original/{}'.format(args.save_num)
if not os.path.exists(save_location):
    os.makedirs(save_location)

save_interval  = 1000
print_interval = 1

n_episodes   = args.n_episodes
sample_len   = 480
obs_data_len = 192
step_len     = 1

risk_aversion_multiplier = 0.5 + args.risk_aversion / 2

n_action_intervals = 5

init_budget = 1

torch.save(hyperparams, os.path.join(save_location, "hyperparams.pth"))

df = pd.read_hdf('../../dataset/binance_data_train.h5', 'STW')
df.fillna(method='ffill', inplace=True)

def main(params):

    env = TradingEnv(custom_args=args, env_id='custom_trading_env', obs_data_len=obs_data_len, step_len=step_len, sample_len=sample_len,
                           df=df, fee=0.001, initial_budget=1, n_action_intervals=n_action_intervals, deal_col_name='c', sell_at_end=True,
                           feature_names=['o', 'h','l','c','v',
                                          'num_trades', 'taker_base_vol'])
    agent = dqn_agent.Agent(action_size=2 * n_action_intervals + 1, obs_len=obs_data_len, num_features=env.reset().shape[-1], \
        buffer_size=params['buffer_size'], n_multisteps =params['n_multisteps'], a=params['a'], v_max=params['v_max'],v_min=params['v_min'], \
        n_atom = params['n_atom'], initial_sigma=params['initial_sigma'])

    beta = 0.4
    beta_inc = (1 - beta) / 1000
    agent.beta = beta

    scores_list = []
    
    
    # for n_epi in range(10000):  # Progress 10,000 rounds 
    for n_epi in range(1,n_episodes+1):
        state = env.reset()
        score = 0.
        actions = []
        rewards = []

        while True:
            action = int(agent.act(state, eps=0.))
            actions.append(action)
            next_state, reward, done, info  = env.step(action)

            rewards.append(reward)
            score += reward
            if reward < 0:
                reward *= risk_aversion_multiplier
            if done:
                action = 2 * n_action_intervals
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
            print_str = "# of episode: {:d}, avg score: {:.4f}\n  Actions: {} fee_rate: {:.6f}".format(n_epi, sum(scores_list[-print_interval:]) / print_interval, np.array(actions), info['fee_rate'])
            print(print_str)
            with open(os.path.join(save_location, "output_log.txt"), mode='a') as f:
                f.write(print_str + '\n')

        if n_epi % save_interval == 0:
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_location, 'TradingGym_Rainbow_{:d}.pth'.format(n_epi)))
            torch.save(scores_list, os.path.join(save_location, 'scores.pth'))
        nni.report_final_result(sum(scores_list)/len(scores_list))
    del env


if __name__ == '__main__':
    try :
        tuner_params = nni.get_next_parameter() # return dictionary of parameters
        logger.debug(tuner_params)
        params = dict( (k , hyperparams[k]) \
            for k in ['buffer_size', 'n_multisteps', 'a', 'v_min', 'v_max', 'n_atoms', 'initial_sigma']) # make dictiionary from instance
        params.update(tuner_params) #set parmater to next parameters
        main(params)
    except Exception as e:
        logging.exception(e)
        raise