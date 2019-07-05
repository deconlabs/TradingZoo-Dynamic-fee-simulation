import numpy as np
import torch
from torch.distributions import Categorical
# device   = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
# def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
#     img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
#     return img


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

# if frame_idx % 1000 == 0 : # 1000번 마다 plot 그려주기
#         print(frame_idx)
#         torch.save(model.state_dict(),'weight/pong_{}.pt'.format(frame_idx+load_weight_n))

#         test_reward = np.mean([test_env() for _ in range(10)])
#         test_rewards.append(test_reward)
#         # plot(frame_idx, test_rewards)
#         print("test_reward : ", np.mean(test_rewards))
#         if test_reward > threshold_reward: early_stop = True
# log_probs = []
#     values    = []
#     states    = []
#     actions   = []
#     rewards   = []
#     masks     = []
#     next_states = []