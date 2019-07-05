# import logging
# log_file_name = "ppo_pong.log"
# logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level = logging.DEBUG)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import random


import gym
import numpy as np
import scipy.signal as signal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


import pong_utils
device   = pong_utils.device

from common.multiprocessing_env import SubprocVecEnv
from pong_utils import preprocess_single, preprocess_batch
from pong_utils import collect_trajectories
from arguments import argparser

args = argparser()
load_weight_n = args.n

num_envs = 16
env_name = "PongDeterministic-v4"
#Hyper params:
hidden_size      = 32
lr               = 1e-4
num_steps        = 128
mini_batch_size  = 256
ppo_epochs       = 3
threshold_reward = 16

max_frames = 150000
discount = 0.99
epsilon = 0.1
beta = 0.4
early_stop = False
n_updates = 4

frame_idx  = 0
scores_list = []

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=4,
                               kernel_size=6,
                               stride=2,
                               bias = False
                               )
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
#The second convolution layer takes a 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=6,
                               stride=4,
                               )
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
#The third convolution layer takes a 9x9 frame and produces a 7x7 frame
        
        self.lin = nn.Linear(in_features=9 * 9 * 8,
                             out_features=256)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
#A fully connected layer to get logits for ππ
        self.pi = nn.Linear(in_features=256,
                                   out_features=6)
        nn.init.orthogonal_(self.pi.weight, np.sqrt(0.01))  # softmax 없어도 괜찮을까? -> relu 이기 때문에 괜찮다. 음수 안들어간다
#A fully connected layer to get value function
        self.value = nn.Linear(in_features=256,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)
    
    def forward(self, obs):

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
      
        h = h.reshape((-1, 9 * 9 * 8))

        h = F.relu(self.lin(h))
        pi = F.softmax(self.pi(h), dim = 1)
        value = F.relu(self.value(h))

        return pi, value


def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    f1 = env.reset()
    f2,_,_,_ = env.step(0)
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = preprocess_batch([f1,f2])
        pi, _ = model(state)
        dist = Categorical(pi)
        f1, r1, done, _ = env.step(dist.sample().cpu().numpy()[0])
        f2, r2, done, _ = env.step(dist.sample().cpu().numpy()[0])
        reward = r1 + r2

        if vis: env.render()
        total_reward += reward
        # print(total_reward)
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage): # 전체 배치에서 mini_batch 를 만드는 것이다.
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i]], log_probs[ids[i]], returns[ids[i], :], advantage[ids[i], :]
        

def ppo_update(ppo_epochs, mini_batch_size, states, actios, log_probs, returns, advantages, clip_param=0.2): # training
    mean_loss = 0
    for _ in range(ppo_epochs):
        loss_sum = 0
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            with torch.autograd.detect_anomaly():
                pi, value = model(state)
                dist = Categorical(pi)
                #     new_log_probs a= dist.log_prob(action)
                
                pi_a = pi.gather(1,action.unsqueeze(-1))
                # logging.warning(f'{pi_a} : pi_a')
                new_log_probs = torch.log(pi_a)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage


                actor_loss  = - torch.min(surr1, surr2).mean()
                # critic_loss = (return_.detach() - value).pow(2).mean()
                entropy = dist.entropy()

                # loss = 0.5 * critic_loss + actor_loss  - 0.01 * entropy
                loss = actor_loss  - 0.01 * entropy

                optimizer.zero_grad()
                
                loss.mean().backward()
                
                optimizer.step()
                loss_sum += loss.mean().item()
        mean_loss+=loss_sum

    print(mean_loss / ppo_epochs)
    return mean_loss / ppo_epochs

def states_to_prob(model, states):
    states = torch.stack(states)
    model_input = states.view(-1,*states.shape[-3:])
    pi, _ = model(model_input)
    return pi

def clipped_surrogate(policy, log_old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.2):
    
    rewards_future = signal.lfilter([1], [1, -discount], np.asarray(rewards)[::-1], axis=0)[::-1]
    
    rewards_normalized = rewards_future - np.mean(rewards_future, axis=1, keepdims=True)
    
    old_probs = torch.stack(log_old_probs).data.to(device).exp()
    actions = torch.stack(actions).long().to(device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = new_probs.view(*old_probs.size(), 6).gather(2, actions.unsqueeze(-1)).squeeze(-1)
    
    reweight_factor = new_probs.div(old_probs)
    
    clipped_reweight_factor = torch.clamp(reweight_factor, 1-epsilon, 1+epsilon)
    
    clipped_surrogate = torch.min(reweight_factor.mul(rewards), clipped_reweight_factor.mul(rewards))

    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(clipped_surrogate.add(entropy.mul(beta)))

model = ActorCritic().to(device) #return dist, v
if args.load_weight:
        model.load_state_dict(torch.load(f'PongDeterministic-v4_{load_weight_n}.pth'))
optimizer = optim.Adam(model.parameters(), lr=lr)

f1 = envs.reset()
f2 = envs.step([0]*num_envs)

if __name__ =="__main__":
    while not early_stop and frame_idx < max_frames:
        frame_idx +=1
        print(frame_idx)
        if frame_idx % 100 == 0 :
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
                
    #     #경험 1세트 모은거로 학습하기
    #     #num_step 만큼 진행했을 때 reward 얼마였는지 구하기
    # #     next_state = preprocess_batch(next_state)
    # #     print("next_state shape: ", next_state.shape) # [16, 1, 80,80]
    #     _, next_value = model(next_state)
    # #     print("next_vlaue shape: " , next_value.shape)
    #     returns = signal.lfilter([1], [1, -discount], np.asarray(rewards)[::-1], axis=0)[::-1]
    #     advantage = returns - np.mean(returns, axis=1, keepdims=True)

    # #     returns = compute_gae(next_value, rewards, masks, values)
    # #     logging.debug(f"returns {returns} and shape is {len(returns)}, {len(returns[0])}" )
    # #     returns = torch.cat(returns).detach()
    #     returns = torch.from_numpy(np.ascontiguousarray(returns)).float().to(device)
    #     advantage = torch.from_numpy(advantage).float().to(device)
    # #     logging.debug("after")
    # #     logging.debug(f"{returns} and shape is {returns.shape}" )
    # #     print(returns.shape, "what's happening here?")
    #     log_probs = torch.cat(log_probs).detach()
    #     values    = torch.cat(values).detach()
    #     states    = torch.cat(states)
    #     actions   = torch.cat(actions)
    # #     advantage = returns - values
        

    #     loss_ = ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns.view(-1, 1), advantage.view(-1, 1))
    #     losses.append(loss_)
        # print(loss)
        if frame_idx % 100 == 0:
                torch.save(model.state_dict(), f'PongDeterministic-v4_{frame_idx+load_weight_n}.pth')
                plt.plot(scores_list)
                plt.title(f'{frame_idx}th score')
                plt.savefig(f'plot/{frame_idx}th_score.png')
                test_rewards = np.mean([test_env() for _ in range(10)])
                print(f"test rewards = {test_rewards}")