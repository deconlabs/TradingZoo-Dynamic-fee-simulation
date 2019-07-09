import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        super().__init__()

        self.lin1 = nn.Linear(in_features=self.observation_space,
                             out_features=512)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))

        nn.BatchNorm1d(batch_size, len(featrue))
        self.lin2 = nn.Linear(in_features=512,
                              out_features=256)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))

#A fully connected layer to get logits for ππ
        self.pi = nn.Linear(in_features=256,
                                   out_features=self.action_space)
        nn.init.orthogonal_(self.pi.weight, np.sqrt(0.01))  # softmax 없어도 괜찮을까? -> relu 이기 때문에 괜찮다. 음수 안들어간다
#A fully connected layer to get value function
        self.value = nn.Linear(in_features=256,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)
    
    def forward(self, obs):

        h = F.relu(self.lin1(obs))
        h = F.relu(self.lin2(obs))

      
        h = h.reshape((-1, ))

        h = F.relu(self.lin(h))
        pi = F.softmax(self.pi(h), dim = 1)
        value = F.relu(self.value(h))

        return pi, value