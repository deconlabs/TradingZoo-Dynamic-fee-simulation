import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import random

import numpy as np
import torch
from torch.distributions import Categorical


import gym
import numpy as np
import scipy.signal as signal


device   = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return torch.from_numpy(img).float().to(device)
# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    # print(list_of_images_prepro.shape)
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    # print(batch_input.shape)
    return torch.from_numpy(batch_input).float().to(device)
    # return torch.from_numpy(list_of_images_prepro).float().to(device)
def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        
        return env


    return _thunk

def collect_trajectories(envs, model, num_steps):
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    # next_states = []
    f1 = envs.reset()
    f2 , _, _ ,_  = envs.step([0]*len(envs.ps))
    
    for _ in range(num_steps): #경험 모으기 - gpu 쓰는구나 . 하지만 여전히 DataParallel 들어갈 여지는 없어보인다. 
        #-> 아.. state 가 벡터 1개가 아닐 것 같다.. 16개네. gpu 쓸만하다. DataParallel 도 가능할듯?
        
        state = preprocess_batch([f1,f2])
        
        pi, value = model(state)
        # print(value, value.shape , "value")
        dist = Categorical(pi)
        action = dist.sample()
        
        f1, r1, done, _ = envs.step(action.cpu().numpy()) 
        f2, r2, done, _ = envs.step(action.cpu().numpy()) 
        reward = r1 + r2
        # print(torch.tensor(reward).sum(dim=0).mean())
        # logging.warning(f'dist[action] : {dist[action]}')
        # print(action)
        log_prob = dist.log_prob(action) #torch.log(dist[action])
        log_probs.append(log_prob) #num_envs*1
        values.append(value) # num_envs * 1
        # rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        rewards.append(reward)
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        states.append(state) #num_envs* 2 * 80 * 80
        actions.append(action) #num_envs*1
        
        next_state = preprocess_batch([f1, f2])
        # next_states.append(next_state)
        state = next_state
        # frame_idx += 1
        if done.any():
            break
    return log_probs , states, actions, rewards , next_state, masks, values

def test_env(env_name, model, vis=False):
    env = gym.make(env_name)
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
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(clipped_surrogate.add(entropy.mul(beta)))