import time
import random
import numpy as np
import pandas as pd
import torch

from custom_trading_env import TradingEnv
import DQNTradingAgent.dqn_agent as dqn_agent
from custom_hyperparameters import hyperparams
from arguments import argparser

args = argparser()

#hyperparmeter
device = torch.device("cpu")
dqn_agent.set_device(device)
load_weight_n = 500

df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')

env = TradingEnv(custom_args=args, env_id='custom_trading_env', obs_data_len=obs_data_len, step_len=step_len, sample_len=sample_len,
                           df=df, fee=0.001, initial_budget=1, n_action_intervals=n_action_intervals, deal_col_name='c', sell_at_end=True,
                           feature_names=['o', 'h','l','c','v',
                                          'num_trades', 'taker_base_vol'])
state = env.reset()
env.render()

agents=[]
for i in range(args.n_agent):
    agent = dqn_agent.Agent(action_size=3, num_features=16, **hyperparams)
    agent.qnetwork_local.load_state_dict(torch.load(f'./saves/{i+1}.pth', map_location=device))
    agent.qnetwork_local.to(device)
    agent.qnetwork_local.eval()
    agents.append(agent)

### randow choice action and show the transaction detail
done = False
while not done:

    action = int(agent.act(state))
    state, reward, done, info = env.step(action)
    env.render()
    if done:
        input()
        break
# env.transaction_details
