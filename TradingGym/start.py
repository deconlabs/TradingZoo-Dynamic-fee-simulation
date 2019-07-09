import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import trading_env
from utils import collect_trajectories, device, clipped_surrogate
from PPOTradingAgent.model import CNNTradingAgent
from PPO.common.multiprocessing_env import  SubprocVecEnv
from arguments import argparser


# Hyperparameters
args = argparser()
device = device
learning_rate = 0.001
discount = 0.995
eps = 0.05
K_epoch = 3
num_steps = 128
beta = 0.4
num_envs = 16

df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df.fillna(method='ffill', inplace=True)
# df.Price = np.arange(len(df))*100.
def make_env():
    def _thunk():
        env = trading_env.make(custom_args= args, env_id='training_v1', obs_data_len=256, step_len=16,
                               df=df, fee=0.0, max_position=5, deal_col_name='Price',
                               feature_names=['Price', 'Volume',
                                              'Ask_price', 'Bid_price',
                                              'Ask_deal_vol', 'Bid_deal_vol',
                                              'Bid/Ask_deal', 'Updown'])

        return env

    return _thunk

def main():
    global beta
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    save_interval = 100

    envs = [make_env() for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)
    model = CNNTradingAgent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print_interval = 10

    scores_list = []
    loss_list = []
    for n_epi in range(10000):  # 게임 1만판 진행
        n_epi +=1
        loss = 0.0
        log_probs, states, actions, rewards, next_state, masks, values = collect_trajectories(envs,model,num_steps)
        
        # raise Exception("True" if torch.any(torch.isnan(torch.stack(states))) else "False")
        if beta>0.01:
            beta*=discount
        for _ in range(K_epoch):
            L = -clipped_surrogate(envs,model, log_probs, states, actions, rewards, discount, eps, beta)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

            loss+=L.item()
            del L

        
        score = np.asarray(rewards).sum()
        scores_list.append(score)
        loss_list.append(loss)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.4f}, loss : {:.6f}".format(
                n_epi, score / print_interval, loss / print_interval))
            print("actions : ", torch.cat(actions))
        
        if n_epi % 100 ==0:
            torch.save(model.state_dict(), f'TradingGym_{n_epi}.pth')
            # print("score_lsit " ,scores_list)
            # print("loss_lsit" ,loss_list)
            # plt.plot(scores_list)
            # plt.title(f'{n_epi}th score')
            # plt.savefig(f'plot/{n_epi}th_score.png')
            # test_rewards = np.mean([test_env(env_name, model) for _ in range(10)])
            # print(f"test rewards = {test_rewards}")
            # print("rewards : ", rewards)

        # if n_epi % save_interval == 0 and n_epi != 0:
        #     if len(os.listdir('minimalRL/weight')) >= 5:
        #         # files = os.listdir('minimalRL/weight')
        #         files = glob.glob("minimalRL/weight/*.pth")
        #         files = list(map(os.path.basename, files))
        #         files.sort(key=lambda x: int(x[12:-4]), reverse=True)

        #         for files in files[5:]:
        #             os.remove(os.path.join('minimalRL/weight/', files))
        #     torch.save(model.state_dict(),
        #                'minimalRL/weight/{}_{}.pth'.format(env_name, n_epi))

    envs.close()


if __name__ == '__main__':
    main()
