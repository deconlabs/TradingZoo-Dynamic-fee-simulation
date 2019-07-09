import time
import random
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

import trading_env
from PPOTradingAgent.model import CNNTradingAgent

#hyperparmeter
device = "cpu"
load_weight_n = 2700

df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')

env = trading_env.make(env_id='training_v1', obs_data_len=256, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price',
                       feature_names=['Price', 'Volume',
                                      'Ask_price','Bid_price',
                                      'Ask_deal_vol','Bid_deal_vol',
                                      'Bid/Ask_deal', 'Updown'])
state = env.reset()
env.render()

model = CNNTradingAgent().to(device)
model.load_state_dict(torch.load(f'TradingGym_{load_weight_n}.pth', map_location=device))
model.eval()

### randow choice action and show the transaction detail
done = False
while not done:

    probs = model(torch.from_numpy(state).float().view(1,256,16).to(device))
    dist = Categorical(probs)
    action = dist.sample()
    state, reward, done, info = env.step(action.cpu().numpy())
    # print(state, reward)
    # env.render()
    #
    # time.sleep(0.3)
    if done:
        input()
        break
    # if (i+1) % 100==0:
    #     input()
# env.transaction_details