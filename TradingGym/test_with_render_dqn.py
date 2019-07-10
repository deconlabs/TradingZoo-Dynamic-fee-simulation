import time
import random
import numpy as np
import pandas as pd
import torch

import trading_env
import DQNTradingAgent.dqn_agent as dqn_agent
from hyperparameters import hyperparams
from arguments import argparser

args = argparser()

#hyperparmeter
device = torch.device("cpu")
dqn_agent.set_device(device)
load_weight_n = 1300

df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')

# env = trading_env.make(custom_args=args, env_id='training_v1', obs_data_len=256, step_len=16,
#                        df=df, fee=0.0, max_position=5, deal_col_name='Price',
#                        feature_names=['Price', 'Volume',
#                                       'Ask_price','Bid_price',
#                                       'Ask_deal_vol','Bid_deal_vol',
#                                       'Bid/Ask_deal', 'Updown'])
env = trading_env.make(custom_args = args, env_id='backtest_v1', obs_data_len=256, step_len=16,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price',
                        feature_names=['Price', 'Volume',
                                       'Ask_price','Bid_price',
                                       'Ask_deal_vol','Bid_deal_vol',
                                       'Bid/Ask_deal', 'Updown'])
state = env.reset()
env.render()

agent = dqn_agent.Agent(action_size=3, num_features=16, **hyperparams)
agent.qnetwork_local.load_state_dict(torch.load('./DQN_with_fee_logs_and_saves/TradingGym_rainbow_{:d}.pth'.format(load_weight_n), map_location=device))
agent.qnetwork_local.to(device)
agent.qnetwork_local.eval()

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
