# import logging
# log_file_name = "ppo_pong.log"
# logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level = logging.DEBUG)
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from model import ActorCritic
import pong_utils
device   = pong_utils.device
from pong_utils import preprocess_single, preprocess_batch, make_env, collect_trajectories, test_env, states_to_prob, clipped_surrogate
from arguments import argparser
from common.multiprocessing_env import SubprocVecEnv

args = argparser()
load_weight_n = args.n

num_envs = 16
env_name = args.env_name
#Hyper params:
hidden_size      = 32
lr               = 1e-4
num_steps        = 128
mini_batch_size  = 256
ppo_epochs       = 3
# threshold_reward = 16

max_episodes = 150000
discount = 0.99
epsilon = 0.1
beta = 0.4
early_stop = False
n_updates = 4
episode_idx  = 0
scores_list = []

if __name__ =="__main__":
    envs = [make_env(env_name) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    model = ActorCritic().to(device) #return dist, v

    if args.load_weight:
            model.load_state_dict(torch.load(f'PongDeterministic-v4_{load_weight_n}.pth'))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    f1 = envs.reset()
    f2 = envs.step([0]*num_envs)
    
    while not early_stop and episode_idx < max_episodes:
        episode_idx +=1
        if episode_idx % 100 == 0 :
            num_steps += args.additional_num_step
        log_probs, states, actions, rewards, next_state, masks, values = collect_trajectories(envs,model,num_steps)
        scores = np.asarray(rewards).sum(axis=0)
        scores_list.append(scores.mean())
        print("Mean:", scores.mean(), "\nRaw:", scores)

            # stop if any of the trajectories is done
            # we want all the lists to be retangular

        for _ in range(n_updates):

            # uncomment to utilize your own clipped function!
            # raise Exception(type(states), states[0].size())
            if args.beta_decay and beta > 0.01:
                beta *= discount
            L = -clipped_surrogate(model, log_probs, states, actions, rewards, discount, epsilon, beta)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            loss = L.item()
            del L
        
        epsilon *= 0.999
        beta *= 0.995
                
        if episode_idx % 100 == 0:
            torch.save(model.state_dict(), f'PongDeterministic-v4_{episode_idx+load_weight_n}.pth')
            plt.plot(scores_list)
            plt.title(f'{episode_idx}th score')
            plt.savefig(f'plot/{episode_idx}th_score.png')
            test_rewards = np.mean([test_env(env_name, model) for _ in range(10)])
            print(f"test rewards = {test_rewards}")