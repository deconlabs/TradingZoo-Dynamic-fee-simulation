# PPOTradingAgent
## What is PPO?
It's a policy-gradient RL algorithm which shows stable performance imporovement. It has 2 notable traits. First, it doesn't use experience which changes network too much. It clips change which means clipped experience has no gradient thus not updating the network. Second, PPO is an on-policy algorithm, which means it doesn't use previous experience which was gained from outdated network. Instead, it collects new experience in every update.

If you want to know more PPO, please watch this less than 20 min [video](https://youtu.be/5P7I-xPq8u8)

## Training Origininal Agent
```python3
python ppo_start.py
```
### you can enjoy with pretrained model with Rendering
```python3
python test_with_render_dqn.py
```

## Transfer Learning 
Agents have differnet risk aversion rate
```shell
bash transfer_learn.sh
```

### Below shows flow of Learning process of PPO agent

```python
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
```


```python
df = pd.read_hdf('dataset/SGXTWsample.h5', 'STW')
df.fillna(method='ffill', inplace=True)
```


```python
# Hyperparameters
class args:
    def __init__(self,no_short):
        self.no_short = no_short
args = args(True)
device = device
learning_rate = 0.001
discount = 0.995
eps = 0.05
K_epoch = 3
num_steps = 128
beta = 0.4
num_envs = 16
```


```python
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
```


```python

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
for n_epi in range(10000):  # Progress 10,000 rounds
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


    score = np.asarray(rewards).sum(axis=0).mean()
    scores_list.append(score)
    loss_list.append(loss)

    if n_epi % print_interval == 0 and n_epi != 0:
        print("# of episode :{}, avg score : {:.4f}, loss : {:.6f}".format(
            n_epi, score / print_interval, loss / print_interval))
        print("actions : ", torch.cat(actions))
        

    if n_epi % save_interval ==0:
        torch.save(model.state_dict(), f'TradingGym_{n_epi}.pth')
        torch.save(scores_list, f"plot/{n_epi}_scores.pth")
        plt.plot(scores_list)
        plt.title("Reward")
        plt.grid(True)
        plt.savefig(f'plot/{n_epi}_ppo.png')
        plt.close()

envs.close()

```

    [2019-07-11 13:56:22,308] Making new env: training_v1
    [2019-07-11 13:56:22,313] Making new env: training_v1
    [2019-07-11 13:56:22,317] Making new env: training_v1
    [2019-07-11 13:56:22,323] Making new env: training_v1
    [2019-07-11 13:56:22,328] Making new env: training_v1
    [2019-07-11 13:56:22,333] Making new env: training_v1
    [2019-07-11 13:56:22,338] Making new env: training_v1
    [2019-07-11 13:56:22,343] Making new env: training_v1
    [2019-07-11 13:56:22,348] Making new env: training_v1
    [2019-07-11 13:56:22,353] Making new env: training_v1
    [2019-07-11 13:56:22,364] Making new env: training_v1
    [2019-07-11 13:56:22,370] Making new env: training_v1
    [2019-07-11 13:56:22,358] Making new env: training_v1
    [2019-07-11 13:56:22,376] Making new env: training_v1
    [2019-07-11 13:56:22,382] Making new env: training_v1
    [2019-07-11 13:56:22,388] Making new env: training_v1
    /home/jeffrey/Binanace_trading_simulation/TradingGym/


    # of episode :10, avg score : -0.0824, loss : -0.125099
    actions :  tensor([1, 1, 2,  ..., 0, 1, 1], device='cuda:2')
    # of episode :20, avg score : 0.1651, loss : -0.119348
    actions :  tensor([2, 1, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :30, avg score : -0.1132, loss : -0.113250
    actions :  tensor([2, 1, 0,  ..., 2, 2, 1], device='cuda:2')
    # of episode :40, avg score : -0.0331, loss : -0.108172
    actions :  tensor([1, 1, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :50, avg score : 0.1204, loss : -0.102430
    actions :  tensor([2, 0, 2,  ..., 2, 0, 0], device='cuda:2')
    # of episode :60, avg score : -0.1065, loss : -0.097657
    actions :  tensor([0, 1, 0,  ..., 0, 2, 1], device='cuda:2')
    # of episode :70, avg score : 0.1535, loss : -0.092856
    actions :  tensor([0, 0, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :80, avg score : 0.1595, loss : -0.088346
    actions :  tensor([1, 1, 1,  ..., 2, 2, 2], device='cuda:2')
    # of episode :90, avg score : -0.0102, loss : -0.084171
    actions :  tensor([2, 0, 0,  ..., 2, 1, 1], device='cuda:2')
    # of episode :100, avg score : 0.0075, loss : -0.080245
    actions :  tensor([2, 1, 0,  ..., 0, 0, 2], device='cuda:2')
    # of episode :110, avg score : -0.0200, loss : -0.076196
    actions :  tensor([2, 0, 1,  ..., 2, 2, 2], device='cuda:2')
    # of episode :120, avg score : -0.0673, loss : -0.072024
    actions :  tensor([1, 0, 2,  ..., 2, 0, 2], device='cuda:2')
    # of episode :130, avg score : -0.0613, loss : -0.068936
    actions :  tensor([0, 1, 2,  ..., 1, 0, 0], device='cuda:2')
    # of episode :140, avg score : 0.1349, loss : -0.065550
    actions :  tensor([2, 0, 0,  ..., 1, 2, 2], device='cuda:2')
    # of episode :150, avg score : 0.1633, loss : -0.062481
    actions :  tensor([0, 2, 0,  ..., 2, 1, 0], device='cuda:2')
    # of episode :160, avg score : -0.0833, loss : -0.059381
    actions :  tensor([2, 2, 0,  ..., 2, 1, 0], device='cuda:2')
    # of episode :170, avg score : -0.0311, loss : -0.056452
    actions :  tensor([1, 1, 1,  ..., 1, 0, 2], device='cuda:2')
    # of episode :180, avg score : -0.0580, loss : -0.053845
    actions :  tensor([0, 2, 2,  ..., 0, 2, 2], device='cuda:2')
    # of episode :190, avg score : -0.0710, loss : -0.050911
    actions :  tensor([2, 0, 0,  ..., 0, 1, 0], device='cuda:2')
    # of episode :200, avg score : 0.0142, loss : -0.048710
    actions :  tensor([1, 1, 1,  ..., 0, 1, 0], device='cuda:2')
    # of episode :210, avg score : 0.1539, loss : -0.046370
    actions :  tensor([0, 0, 1,  ..., 0, 0, 1], device='cuda:2')
    # of episode :220, avg score : -0.1098, loss : -0.043918
    actions :  tensor([2, 0, 0,  ..., 1, 2, 0], device='cuda:2')
    # of episode :230, avg score : -0.0150, loss : -0.041735
    actions :  tensor([2, 2, 2,  ..., 0, 2, 1], device='cuda:2')
    # of episode :240, avg score : 0.1486, loss : -0.039882
    actions :  tensor([2, 1, 1,  ..., 0, 1, 1], device='cuda:2')
    # of episode :250, avg score : 0.0005, loss : -0.037948
    actions :  tensor([2, 1, 2,  ..., 2, 1, 0], device='cuda:2')
    # of episode :260, avg score : 0.1490, loss : -0.035955
    actions :  tensor([2, 2, 1,  ..., 0, 1, 2], device='cuda:2')
    # of episode :270, avg score : -0.0873, loss : -0.034252
    actions :  tensor([0, 1, 1,  ..., 1, 1, 1], device='cuda:2')
    # of episode :280, avg score : -0.0399, loss : -0.032322
    actions :  tensor([1, 2, 1,  ..., 2, 0, 0], device='cuda:2')
    # of episode :290, avg score : -0.0308, loss : -0.030816
    actions :  tensor([1, 0, 2,  ..., 2, 2, 0], device='cuda:2')
    # of episode :300, avg score : -0.1399, loss : -0.029301
    actions :  tensor([1, 0, 1,  ..., 2, 0, 0], device='cuda:2')
    # of episode :310, avg score : -0.0993, loss : -0.027952
    actions :  tensor([0, 1, 1,  ..., 2, 1, 0], device='cuda:2')
    # of episode :320, avg score : 0.1089, loss : -0.026855
    actions :  tensor([0, 1, 1,  ..., 0, 1, 2], device='cuda:2')
    # of episode :330, avg score : -0.1013, loss : -0.025549
    actions :  tensor([0, 0, 2,  ..., 1, 0, 0], device='cuda:2')
    # of episode :340, avg score : -0.1051, loss : -0.023983
    actions :  tensor([0, 1, 0,  ..., 0, 0, 0], device='cuda:2')
    # of episode :350, avg score : -0.0543, loss : -0.022752
    actions :  tensor([0, 1, 0,  ..., 2, 0, 2], device='cuda:2')
    # of episode :360, avg score : -0.0791, loss : -0.021587
    actions :  tensor([0, 1, 0,  ..., 2, 0, 0], device='cuda:2')
    # of episode :370, avg score : -0.0752, loss : -0.020728
    actions :  tensor([0, 2, 0,  ..., 1, 2, 0], device='cuda:2')
    # of episode :380, avg score : -0.0320, loss : -0.019515
    actions :  tensor([2, 2, 0,  ..., 0, 2, 1], device='cuda:2')
    # of episode :390, avg score : -0.1021, loss : -0.018811
    actions :  tensor([0, 1, 1,  ..., 0, 1, 1], device='cuda:2')
    # of episode :400, avg score : -0.0365, loss : -0.017905
    actions :  tensor([1, 1, 2,  ..., 1, 0, 1], device='cuda:2')
    # of episode :410, avg score : -0.0906, loss : -0.017157
    actions :  tensor([1, 2, 0,  ..., 0, 0, 0], device='cuda:2')
    # of episode :420, avg score : -0.1021, loss : -0.016012
    actions :  tensor([2, 2, 2,  ..., 1, 1, 2], device='cuda:2')
    # of episode :430, avg score : 0.1236, loss : -0.015087
    actions :  tensor([0, 0, 0,  ..., 2, 2, 2], device='cuda:2')
    # of episode :440, avg score : 0.1454, loss : -0.014756
    actions :  tensor([2, 2, 1,  ..., 0, 2, 2], device='cuda:2')
    # of episode :450, avg score : -0.0224, loss : -0.013832
    actions :  tensor([1, 2, 1,  ..., 1, 2, 0], device='cuda:2')
    # of episode :460, avg score : -0.1046, loss : -0.013287
    actions :  tensor([0, 1, 2,  ..., 1, 0, 0], device='cuda:2')
    # of episode :470, avg score : 0.1152, loss : -0.012341
    actions :  tensor([1, 2, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :480, avg score : 0.0732, loss : -0.012284
    actions :  tensor([2, 0, 0,  ..., 2, 1, 0], device='cuda:2')
    # of episode :490, avg score : 0.0014, loss : -0.011767
    actions :  tensor([0, 0, 1,  ..., 1, 2, 2], device='cuda:2')
    # of episode :500, avg score : -0.0335, loss : -0.011069
    actions :  tensor([2, 2, 0,  ..., 1, 1, 0], device='cuda:2')
    # of episode :510, avg score : -0.1118, loss : -0.010298
    actions :  tensor([0, 1, 0,  ..., 1, 0, 2], device='cuda:2')
    # of episode :520, avg score : -0.0188, loss : -0.009145
    actions :  tensor([1, 1, 1,  ..., 0, 0, 1], device='cuda:2')
    # of episode :530, avg score : -0.0791, loss : -0.008669
    actions :  tensor([2, 1, 0,  ..., 0, 1, 2], device='cuda:2')
    # of episode :540, avg score : -0.0239, loss : -0.008668
    actions :  tensor([1, 2, 1,  ..., 2, 1, 0], device='cuda:2')
    # of episode :550, avg score : 0.0222, loss : -0.008534
    actions :  tensor([0, 2, 2,  ..., 2, 2, 0], device='cuda:2')
    # of episode :560, avg score : 0.0887, loss : -0.006650
    actions :  tensor([1, 2, 2,  ..., 1, 2, 0], device='cuda:2')
    # of episode :570, avg score : -0.0960, loss : -0.007730
    actions :  tensor([2, 2, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :580, avg score : -0.0452, loss : -0.006959
    actions :  tensor([2, 1, 1,  ..., 1, 1, 1], device='cuda:2')
    # of episode :590, avg score : -0.0322, loss : -0.005121
    actions :  tensor([1, 0, 1,  ..., 1, 1, 2], device='cuda:2')
    # of episode :600, avg score : 0.0712, loss : -0.005396
    actions :  tensor([0, 0, 1,  ..., 2, 0, 0], device='cuda:2')
    # of episode :610, avg score : 0.0588, loss : -0.005569
    actions :  tensor([2, 0, 0,  ..., 0, 2, 2], device='cuda:2')
    # of episode :620, avg score : -0.0293, loss : -0.005745
    actions :  tensor([2, 0, 0,  ..., 0, 0, 0], device='cuda:2')
    # of episode :630, avg score : -0.0821, loss : -0.004966
    actions :  tensor([0, 0, 2,  ..., 2, 2, 0], device='cuda:2')
    # of episode :640, avg score : -0.0348, loss : -0.003920
    actions :  tensor([2, 1, 1,  ..., 2, 2, 2], device='cuda:2')
    # of episode :650, avg score : -0.0255, loss : -0.004672
    actions :  tensor([2, 1, 1,  ..., 1, 2, 2], device='cuda:2')
    # of episode :660, avg score : -0.0165, loss : -0.004681
    actions :  tensor([0, 2, 0,  ..., 2, 0, 2], device='cuda:2')
    # of episode :670, avg score : -0.0474, loss : -0.003645
    actions :  tensor([0, 1, 0,  ..., 0, 0, 0], device='cuda:2')
    # of episode :680, avg score : 0.0500, loss : -0.004132
    actions :  tensor([2, 2, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :690, avg score : -0.0136, loss : -0.004152
    actions :  tensor([2, 2, 0,  ..., 1, 0, 2], device='cuda:2')
    # of episode :700, avg score : -0.0084, loss : -0.003813
    actions :  tensor([0, 1, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :710, avg score : -0.0383, loss : -0.003238
    actions :  tensor([0, 2, 0,  ..., 0, 2, 2], device='cuda:2')
    # of episode :720, avg score : 0.0582, loss : -0.003575
    actions :  tensor([2, 1, 0,  ..., 0, 0, 0], device='cuda:2')
    # of episode :730, avg score : 0.0821, loss : -0.003508
    actions :  tensor([0, 1, 1,  ..., 0, 2, 0], device='cuda:2')
    # of episode :740, avg score : -0.0076, loss : -0.003328
    actions :  tensor([1, 1, 2,  ..., 0, 0, 1], device='cuda:2')
    # of episode :750, avg score : -0.1140, loss : -0.002735
    actions :  tensor([0, 1, 1,  ..., 0, 1, 1], device='cuda:2')
    # of episode :760, avg score : -0.0424, loss : -0.002440
    actions :  tensor([0, 2, 0,  ..., 1, 0, 2], device='cuda:2')
    # of episode :770, avg score : -0.0494, loss : -0.003053
    actions :  tensor([0, 1, 0,  ..., 0, 1, 0], device='cuda:2')
    # of episode :780, avg score : -0.0453, loss : -0.003249
    actions :  tensor([0, 1, 2,  ..., 2, 0, 1], device='cuda:2')
    # of episode :790, avg score : -0.0367, loss : -0.003179
    actions :  tensor([0, 2, 1,  ..., 2, 2, 0], device='cuda:2')
    # of episode :800, avg score : -0.0482, loss : -0.003050
    actions :  tensor([2, 1, 2,  ..., 1, 0, 0], device='cuda:2')
    # of episode :810, avg score : -0.0233, loss : -0.002960
    actions :  tensor([1, 0, 0,  ..., 0, 0, 2], device='cuda:2')
    # of episode :820, avg score : -0.0207, loss : -0.003256
    actions :  tensor([0, 0, 0,  ..., 2, 1, 2], device='cuda:2')
    # of episode :830, avg score : -0.0863, loss : -0.002718
    actions :  tensor([1, 2, 2,  ..., 2, 2, 0], device='cuda:2')
    # of episode :840, avg score : 0.0529, loss : -0.002985
    actions :  tensor([0, 2, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :850, avg score : 0.0084, loss : -0.003326
    actions :  tensor([0, 0, 0,  ..., 1, 2, 2], device='cuda:2')
    # of episode :860, avg score : -0.0496, loss : -0.003326
    actions :  tensor([1, 1, 0,  ..., 2, 2, 0], device='cuda:2')
    # of episode :870, avg score : 0.0797, loss : -0.002972
    actions :  tensor([0, 0, 0,  ..., 2, 1, 2], device='cuda:2')
    # of episode :880, avg score : 0.0607, loss : -0.003356
    actions :  tensor([0, 2, 2,  ..., 2, 1, 0], device='cuda:2')
    # of episode :890, avg score : 0.0083, loss : -0.003813
    actions :  tensor([2, 0, 2,  ..., 0, 2, 0], device='cuda:2')
    # of episode :900, avg score : -0.0959, loss : -0.003151
    actions :  tensor([0, 2, 2,  ..., 2, 1, 0], device='cuda:2')
    # of episode :910, avg score : -0.0408, loss : -0.003011
    actions :  tensor([2, 2, 1,  ..., 2, 1, 1], device='cuda:2')
    # of episode :920, avg score : -0.0168, loss : -0.003215
    actions :  tensor([0, 1, 2,  ..., 0, 1, 2], device='cuda:2')
    # of episode :930, avg score : -0.0250, loss : -0.003415
    actions :  tensor([0, 2, 0,  ..., 0, 2, 1], device='cuda:2')
    # of episode :940, avg score : -0.0191, loss : -0.003033
    actions :  tensor([0, 0, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :950, avg score : -0.0576, loss : -0.003202
    actions :  tensor([0, 0, 1,  ..., 1, 2, 1], device='cuda:2')
    # of episode :960, avg score : 0.0560, loss : -0.002872
    actions :  tensor([1, 0, 2,  ..., 2, 2, 1], device='cuda:2')
    # of episode :970, avg score : 0.0020, loss : -0.003064
    actions :  tensor([1, 1, 1,  ..., 1, 1, 2], device='cuda:2')
    # of episode :980, avg score : 0.0456, loss : -0.003297
    actions :  tensor([1, 0, 0,  ..., 0, 1, 0], device='cuda:2')
    # of episode :990, avg score : 0.0137, loss : -0.003232
    actions :  tensor([1, 0, 0,  ..., 0, 2, 2], device='cuda:2')
    # of episode :1000, avg score : -0.0286, loss : -0.003325
    actions :  tensor([2, 1, 2,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1010, avg score : -0.0083, loss : -0.003331
    actions :  tensor([2, 0, 1,  ..., 0, 0, 1], device='cuda:2')
    # of episode :1020, avg score : -0.0600, loss : -0.002737
    actions :  tensor([2, 2, 0,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1030, avg score : 0.0102, loss : -0.003189
    actions :  tensor([2, 2, 2,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1040, avg score : 0.0837, loss : -0.003172
    actions :  tensor([1, 0, 2,  ..., 0, 0, 0], device='cuda:2')
    # of episode :1050, avg score : 0.0159, loss : -0.002893
    actions :  tensor([0, 2, 1,  ..., 0, 0, 0], device='cuda:2')
    # of episode :1060, avg score : -0.0527, loss : -0.001911
    actions :  tensor([1, 0, 1,  ..., 1, 1, 2], device='cuda:2')
    # of episode :1070, avg score : -0.0306, loss : -0.001496
    actions :  tensor([2, 1, 2,  ..., 2, 1, 2], device='cuda:2')
    # of episode :1080, avg score : 0.0422, loss : -0.000367
    actions :  tensor([2, 2, 2,  ..., 0, 2, 2], device='cuda:2')
    # of episode :1090, avg score : 0.0385, loss : -0.001904
    actions :  tensor([0, 0, 2,  ..., 1, 1, 2], device='cuda:2')
    # of episode :1100, avg score : 0.0422, loss : -0.002921
    actions :  tensor([2, 1, 2,  ..., 0, 0, 0], device='cuda:2')
    # of episode :1110, avg score : 0.0516, loss : -0.003347
    actions :  tensor([0, 0, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :1120, avg score : -0.0325, loss : -0.003343
    actions :  tensor([1, 0, 0,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1130, avg score : 0.1011, loss : -0.003552
    actions :  tensor([1, 0, 1,  ..., 1, 2, 1], device='cuda:2')
    # of episode :1140, avg score : -0.0372, loss : -0.003379
    actions :  tensor([1, 1, 2,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1150, avg score : 0.1277, loss : -0.003215
    actions :  tensor([0, 2, 0,  ..., 1, 0, 2], device='cuda:2')
    # of episode :1160, avg score : 0.0344, loss : -0.002534
    actions :  tensor([0, 0, 2,  ..., 2, 0, 0], device='cuda:2')
    # of episode :1170, avg score : 0.0990, loss : -0.002804
    actions :  tensor([0, 2, 2,  ..., 2, 0, 2], device='cuda:2')
    # of episode :1180, avg score : 0.0303, loss : -0.003479
    actions :  tensor([2, 0, 0,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1190, avg score : -0.1060, loss : -0.003375
    actions :  tensor([0, 2, 0,  ..., 0, 2, 1], device='cuda:2')
    # of episode :1200, avg score : -0.0666, loss : -0.002954
    actions :  tensor([0, 0, 0,  ..., 0, 0, 1], device='cuda:2')
    # of episode :1210, avg score : -0.0461, loss : -0.002186
    actions :  tensor([0, 0, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :1220, avg score : 0.0674, loss : -0.003031
    actions :  tensor([1, 1, 1,  ..., 2, 2, 0], device='cuda:2')
    # of episode :1230, avg score : 0.0762, loss : -0.002362
    actions :  tensor([0, 1, 0,  ..., 2, 0, 0], device='cuda:2')
    # of episode :1240, avg score : 0.0782, loss : -0.002458
    actions :  tensor([2, 0, 0,  ..., 1, 2, 0], device='cuda:2')
    # of episode :1250, avg score : 0.0826, loss : -0.003150
    actions :  tensor([0, 0, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :1260, avg score : 0.0767, loss : -0.003187
    actions :  tensor([0, 2, 2,  ..., 2, 0, 2], device='cuda:2')
    # of episode :1270, avg score : 0.0136, loss : -0.001223
    actions :  tensor([2, 0, 0,  ..., 1, 2, 2], device='cuda:2')
    # of episode :1280, avg score : -0.0262, loss : -0.002076
    actions :  tensor([0, 0, 2,  ..., 0, 2, 1], device='cuda:2')
    # of episode :1290, avg score : 0.0080, loss : -0.001166
    actions :  tensor([0, 2, 0,  ..., 0, 1, 2], device='cuda:2')
    # of episode :1300, avg score : -0.0398, loss : -0.002663
    actions :  tensor([0, 0, 2,  ..., 0, 2, 2], device='cuda:2')
    # of episode :1310, avg score : 0.0655, loss : -0.003185
    actions :  tensor([2, 2, 0,  ..., 1, 0, 0], device='cuda:2')
    # of episode :1320, avg score : 0.0428, loss : -0.002676
    actions :  tensor([0, 2, 1,  ..., 2, 0, 2], device='cuda:2')
    # of episode :1330, avg score : -0.0462, loss : -0.002735
    actions :  tensor([2, 0, 2,  ..., 0, 0, 0], device='cuda:2')
    # of episode :1340, avg score : -0.0443, loss : -0.003482
    actions :  tensor([0, 0, 2,  ..., 0, 1, 0], device='cuda:2')
    # of episode :1350, avg score : -0.0241, loss : -0.003370
    actions :  tensor([1, 0, 2,  ..., 2, 0, 0], device='cuda:2')
    # of episode :1360, avg score : 0.0690, loss : -0.003288
    actions :  tensor([2, 0, 2,  ..., 2, 2, 2], device='cuda:2')
    # of episode :1370, avg score : 0.0370, loss : -0.003235
    actions :  tensor([2, 0, 0,  ..., 2, 2, 2], device='cuda:2')
    # of episode :1380, avg score : -0.0097, loss : -0.003239
    actions :  tensor([0, 1, 2,  ..., 1, 2, 0], device='cuda:2')
    # of episode :1390, avg score : -0.0214, loss : -0.003216
    actions :  tensor([1, 0, 2,  ..., 0, 1, 2], device='cuda:2')
    # of episode :1400, avg score : -0.0410, loss : -0.001707
    actions :  tensor([0, 2, 2,  ..., 1, 0, 0], device='cuda:2')
    # of episode :1410, avg score : -0.0395, loss : -0.001139
    actions :  tensor([0, 1, 0,  ..., 2, 2, 2], device='cuda:2')
    # of episode :1420, avg score : 0.0061, loss : -0.001629
    actions :  tensor([1, 0, 0,  ..., 0, 2, 0], device='cuda:2')
    # of episode :1430, avg score : 0.0574, loss : -0.003096
    actions :  tensor([0, 1, 2,  ..., 2, 0, 2], device='cuda:2')
    # of episode :1440, avg score : 0.0066, loss : -0.002826
    actions :  tensor([0, 0, 0,  ..., 2, 0, 2], device='cuda:2')
    # of episode :1450, avg score : 0.0114, loss : -0.003004
    actions :  tensor([0, 1, 2,  ..., 1, 1, 2], device='cuda:2')
    # of episode :1460, avg score : -0.0163, loss : -0.002691
    actions :  tensor([2, 0, 1,  ..., 0, 0, 2], device='cuda:2')
    # of episode :1470, avg score : -0.0519, loss : -0.003061
    actions :  tensor([2, 0, 0,  ..., 2, 2, 1], device='cuda:2')
    # of episode :1480, avg score : -0.0232, loss : -0.003041
    actions :  tensor([0, 2, 0,  ..., 2, 0, 2], device='cuda:2')
    # of episode :1490, avg score : -0.0113, loss : -0.002553
    actions :  tensor([1, 0, 1,  ..., 0, 1, 1], device='cuda:2')
    # of episode :1500, avg score : 0.0792, loss : -0.003410
    actions :  tensor([0, 1, 0,  ..., 0, 2, 2], device='cuda:2')
    # of episode :1510, avg score : -0.0664, loss : -0.002555
    actions :  tensor([2, 2, 2,  ..., 0, 0, 0], device='cuda:2')
    # of episode :1520, avg score : -0.0364, loss : -0.003227
    actions :  tensor([0, 0, 2,  ..., 2, 0, 0], device='cuda:2')
    # of episode :1530, avg score : -0.0811, loss : -0.003332
    actions :  tensor([0, 2, 1,  ..., 0, 1, 0], device='cuda:2')
    # of episode :1540, avg score : -0.0660, loss : -0.003401
    actions :  tensor([2, 2, 0,  ..., 2, 1, 1], device='cuda:2')
    # of episode :1550, avg score : -0.0164, loss : -0.003381
    actions :  tensor([2, 0, 1,  ..., 0, 1, 0], device='cuda:2')
    # of episode :1560, avg score : -0.0704, loss : -0.003272
    actions :  tensor([2, 0, 1,  ..., 1, 2, 2], device='cuda:2')
    # of episode :1570, avg score : -0.0710, loss : -0.003242
    actions :  tensor([1, 1, 1,  ..., 2, 1, 1], device='cuda:2')
    # of episode :1580, avg score : -0.0097, loss : -0.003217
    actions :  tensor([0, 2, 0,  ..., 1, 0, 2], device='cuda:2')
    # of episode :1590, avg score : 0.0737, loss : -0.003448
    actions :  tensor([2, 0, 1,  ..., 2, 2, 0], device='cuda:2')
    # of episode :1600, avg score : -0.0092, loss : -0.003453
    actions :  tensor([2, 0, 2,  ..., 0, 2, 2], device='cuda:2')
    # of episode :1610, avg score : 0.1037, loss : -0.003167
    actions :  tensor([2, 0, 0,  ..., 2, 0, 1], device='cuda:2')
    # of episode :1620, avg score : -0.0057, loss : -0.002787
    actions :  tensor([2, 0, 0,  ..., 2, 2, 1], device='cuda:2')
    # of episode :1630, avg score : -0.0585, loss : -0.002309
    actions :  tensor([2, 1, 2,  ..., 2, 2, 1], device='cuda:2')
    # of episode :1640, avg score : -0.0145, loss : -0.002987
    actions :  tensor([2, 0, 2,  ..., 0, 2, 2], device='cuda:2')
    # of episode :1650, avg score : 0.1019, loss : -0.002685
    actions :  tensor([1, 2, 0,  ..., 1, 2, 0], device='cuda:2')
    # of episode :1660, avg score : -0.0550, loss : -0.002749
    actions :  tensor([1, 2, 0,  ..., 1, 0, 2], device='cuda:2')

