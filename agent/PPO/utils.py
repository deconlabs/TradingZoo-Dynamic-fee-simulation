import numpy as np
from scipy import signal

import torch
from torch.distributions import Categorical
from arguments import argparser

# args = argparser()
device = "cuda:0" #"cuda:" + str(args.device_num) if torch.cuda.is_available() else "cpu"

def states_to_prob(model, states):
    states = torch.stack(states)
    model_input = states.view(-1, states.shape[-2], states.shape[-1])
    pi = model(model_input)
    return pi


def collect_trajectories(envs, model, num_steps, ):
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    # next_states = []
    state = envs.reset()

    # buy = True
    for _ in range(num_steps):  
        # probs = model(torch.from_numpy(state).unsqueeze(0).float().to(device))
        # print("state shape", state.shape)
        probs = model(torch.from_numpy(state).float().to(device))
        dist = Categorical(probs)
        action = dist.sample()
 
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        # print("reward after step" , reward)

        log_prob = dist.log_prob(action)  # torch.log(dist[action])
        log_probs.append(log_prob)  # num_envs*1
        # values.append(value)  # num_envs * 1
  
        rewards.append(reward)
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

        states.append(torch.FloatTensor(state).to(device))  # num_envs* 2 * 80 * 80
        actions.append(action)  # num_envs*1

        # next_states.append(next_state)
        state = next_state
        # frame_idx += 1
        if done.any():
            break
    return log_probs, states, actions, rewards, next_state, masks, values


def clipped_surrogate(env,policy, log_old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.2):
    rewards_future = signal.lfilter([1], [1, -discount], np.asarray(rewards)[::-1], axis=0)[::-1]

    rewards_normalized = rewards_future - np.mean(rewards_future, axis=1, keepdims=True)
    # rewards_normalized = np.ascontiguousarray(rewards_future)
    old_probs = torch.stack(log_old_probs).data.to(device).exp()
    actions = torch.stack(actions).long().to(device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    entropy = Categorical(new_probs).entropy().mean()

    # print("new_probs shape ", new_probs.shape)
    new_probs = new_probs.view(*old_probs.size(), env.action_space).gather(2, actions.unsqueeze(-1)).squeeze(-1)

    reweight_factor = new_probs.div(old_probs)

    clipped_reweight_factor = torch.clamp(reweight_factor, 1 - epsilon, 1 + epsilon)

    clipped_surrogate = torch.min(reweight_factor.mul(rewards), clipped_reweight_factor.mul(rewards))

    # include a regularization term
    # entropy = -(new_probs * torch.log(old_probs + 1.e-10) + \
    #             (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    return torch.mean(clipped_surrogate.add(entropy.mul(beta)))