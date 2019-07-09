import random
import numpy as np
import pandas as pd
import trading_env
from PPOTradingAgent.model import CNNTradingAgent
from utils import device

#hyperparmeter
load_weight_n = 100

df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')

env = trading_env.make(env_id='training_v1', obs_data_len=256, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price',
                       feature_names=['Price', 'Volume',
                                      'Ask_price','Bid_price',
                                      'Ask_deal_vol','Bid_deal_vol',
                                      'Bid/Ask_deal', 'Updown'])
env.reset()
env.render()

model = CNNTradingAgent().to(device)
model.load_state_dict(torch.load(f'TradingGym_{load_weight_n}.pth'))
model.eval()
state, reward, done, info = env.step()

### randow choice action and show the transaction detail
for i in range(500):
    i+=1
    probs = model(torch.from_numpy(state).float().to(device))
    dist = Categorical(probs)
    action = dist.sample()
    state, reward, done, info = env.step(action.cpu().numpy())
    print(state, reward)
    env.render()
    if done:
        break
    if (i+1) % 100==0:
        input()
# env.transaction_details