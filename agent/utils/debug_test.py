import random
import numpy as np
import pandas as pd
# import trading_env
from collections import deque
from custom_trading_env import TradingEnv

df = pd.read_hdf('dataset/binance_data.h5', 'STW')

custom_args = object()

obs_data_len = 256
step_len     = 16

_persist_period = obs_data_len // step_len
_reward_deque = deque(maxlen=_persist_period)

# env = trading_env.make(env_id='training_v1', obs_data_len=obs_data_len, step_len=step_len,
#                        df=df, fee=0.1, max_position=5, deal_col_name='Price',
#                        feature_names=['Price', 'Volume',
#                                       'Ask_price','Bid_price',
#                                       'Ask_deal_vol','Bid_deal_vol',
#                                       'Bid/Ask_deal', 'Updown'])

env = TradingEnv(custom_args=custom_args, env_id='trading', obs_data_len=obs_data_len, step_len=step_len,
                 df=df, fee=0.1, initial_budget=1, deal_col_name='c',
                 feature_names=['o', 'h', 'l', 'c', 'v',
                                'num_trades', 'taker_base_vol'], sample_len = 1000, n_action_intervals = 10)

env.reset()
env.render()

hold_buy_sell_probs = [0.34, 0.33, 0.33]
assert sum(hold_buy_sell_probs) == 1, "Probability not summing up to 1"

def get_action(probs):
    action_value = random.random()
    action = 0
    for prob in probs:
        if action_value <= prob:
            break
        else:
            action_value -= prob
            action += 1
    return action

def get_action_input():
    user_input = input("Buy(0-9), Hold(10), or Sell(11-20): ").strip()
    while user_input not in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'):
        user_input = input("Buy(0-9), Hold(10), or Sell(11-20): ").strip()
    return int(user_input)

# state, reward, done, info = env.step(get_action(hold_buy_sell_probs))

### randow choice action and show the transaction detail
for i in range(500):
    print(i)
    # action = get_action(hold_buy_sell_probs)
    action = get_action_input()
    state, reward, done, info = env.step(action)
    print("    Reward:", reward)
    print("budget : ", env.budget)
    env.render()
    if done:
        break
env.transaction_details

input("\nENTER to close")
