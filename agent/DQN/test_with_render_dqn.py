import time
import random
import numpy as np
import pandas as pd
import torch
import os

from custom_trading_env import TradingEnv
import DQNTradingAgent.dqn_agent as dqn_agent
from arguments import argparser

args = argparser()

save_location = f"./saves/{args.agent_num}"

#hyperparmeter
device = torch.device("cpu")
dqn_agent.set_device(device)
load_weight_n = 2000

df = pd.read_hdf('./dataset/binance_data_test.h5', 'STW')
df.fillna(method='ffill', inplace=True)

sample_len   = len(df)
obs_data_len = 192
step_len     = 1

n_action_intervals = 5

init_budget = 1

env = TradingEnv(custom_args=args, env_id='custom_trading_env', obs_data_len=obs_data_len, step_len=step_len, sample_len=sample_len,
                 df=df, fee=0.001, initial_budget=1, n_action_intervals=n_action_intervals, deal_col_name='c', sell_at_end=True,
                 feature_names=['o', 'h','l','c','v',
                                'num_trades', 'taker_base_vol'])


state = env.reset()
env.render()

hyperparams = torch.load(os.path.join(save_location, "hyperparams.pth"))

agent = dqn_agent.Agent(action_size=2 * n_action_intervals + 1, obs_len=obs_data_len, num_features=env.reset().shape[-1], **hyperparams)
agent.qnetwork_local.load_state_dict(torch.load(os.path.join(save_location, 'TradingGym_rainbow_{:d}.pth'.format(load_weight_n)), map_location=device))
agent.qnetwork_local.to(device)
agent.qnetwork_local.eval()

done = False

while not done:
    
    action = int(agent.act(state))
    state, reward, done, info = env.step(action)
    
    print("Action:", action)
    print("Budget:", env.budget)
    print("Reward:", reward)
    if args.render:
        env.render()
    if done:
        break
print(info)
input("Press ENTER to exit>")
