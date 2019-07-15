import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork
from .replay_buffer import ReplayBuffer, rp_set_device
from .default_hyperparameters import SEED, BUFFER_SIZE, BATCH_SIZE, START_SINCE,\
                                    GAMMA, T_UPDATE, TAU, LR, WEIGHT_DECAY, UPDATE_EVERY,\
                                    A, INIT_BETA, P_EPS, N_STEPS, V_MIN, V_MAX,\
                                    CLIP, N_ATOMS, INIT_SIGMA, LINEAR, FACTORIZED

device = torch.device("cpu")

def set_device(new_device):
    global device
    device = new_device
    rp_set_device(new_device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, obs_len, num_features=16, seed=SEED, batch_size=BATCH_SIZE,
                 buffer_size=BUFFER_SIZE, start_since=START_SINCE, gamma=GAMMA, target_update_every=T_UPDATE,
                 tau=TAU, lr=LR, weight_decay=WEIGHT_DECAY, update_every=UPDATE_EVERY, priority_eps=P_EPS,
                 a=A, initial_beta=INIT_BETA, n_multisteps=N_STEPS,
                 v_min=V_MIN, v_max=V_MAX, clip=CLIP, n_atoms=N_ATOMS,
                 initial_sigma=INIT_SIGMA, linear_type=LINEAR, factorized=FACTORIZED, **kwds):
        """Initialize an Agent object.

        Params
        ======
            action_size (int): dimension of each action
            obs_len(int)
            num_features (int): number of features in the state
            seed (int): random seed
            batch_size (int): size of each sample batch
            buffer_size (int): size of the experience memory buffer
            start_since (int): number of steps to collect before start training
            gamma (float): discount factor
            target_update_every (int): how often to update the target network
            tau (float): target network soft-update parameter
            lr (float): learning rate
            weight_decay (float): weight decay for optimizer
            update_every (int): update(learning and target update) interval
            priority_eps (float): small base value for priorities
            a (float): priority exponent parameter
            initial_beta (float): initial importance-sampling weight
            n_multisteps (int): number of steps to consider for each experience
            v_min (float): minimum reward support value
            v_max (float): maximum reward support value
            clip (float): gradient norm clipping (`None` to disable)
            n_atoms (int): number of atoms in the discrete support distribution
            initial_sigma (float): initial noise parameter weights
            linear_type (str): one of ('linear', 'noisy'); type of linear layer to use
            factorized (bool): whether to use factorized gaussian noise in noisy layers
        """
        if kwds != {}:
            print("Ignored keyword arguments: ", end='')
            print(*kwds, sep=', ')
        assert isinstance(action_size, int)
        assert isinstance(obs_len, int)
        assert isinstance(num_features, int)
        assert isinstance(seed, int)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(buffer_size, int) and buffer_size >= batch_size
        assert isinstance(start_since, int) and batch_size <= start_since <= buffer_size
        assert isinstance(gamma, (int, float)) and 0 <= gamma <= 1
        assert isinstance(target_update_every, int) and target_update_every > 0
        assert isinstance(tau, (int, float)) and 0 <= tau <= 1
        assert isinstance(lr, (int, float)) and lr >= 0
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0
        assert isinstance(update_every, int) and update_every > 0
        assert isinstance(priority_eps, (int, float)) and priority_eps >= 0
        assert isinstance(a, (int, float)) and 0 <= a <= 1
        assert isinstance(initial_beta, (int, float)) and 0 <= initial_beta <= 1
        assert isinstance(n_multisteps, int) and n_multisteps > 0
        assert isinstance(v_min, (int, float)) and isinstance(v_max, (int, float)) and v_min < v_max
        if clip: assert isinstance(clip, (int, float)) and clip >= 0
        assert isinstance(n_atoms, int) and n_atoms > 0
        assert isinstance(initial_sigma, (int, float)) and initial_sigma >= 0
        assert isinstance(linear_type, str) and linear_type.strip().lower() in ('linear', 'noisy')
        assert isinstance(factorized, bool)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.action_size         = action_size
        self.obs_len             = obs_len
        self.num_features        = num_features
        self.seed                = seed
        self.batch_size          = batch_size
        self.buffer_size         = buffer_size
        self.start_since         = start_since
        self.gamma               = gamma
        self.target_update_every = target_update_every
        self.tau                 = tau
        self.lr                  = lr
        self.weight_decay        = weight_decay
        self.update_every        = update_every
        self.priority_eps        = priority_eps
        self.a                   = a
        self.beta                = initial_beta
        self.n_multisteps        = n_multisteps
        self.v_min               = v_min
        self.v_max               = v_max
        self.clip                = clip
        self.n_atoms             = n_atoms
        self.initial_sigma       = initial_sigma
        self.linear_type         = linear_type.strip().lower()
        self.factorized          = factorized

        # Distribution
        self.supports = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.delta_z  = (v_max - v_min) / (n_atoms - 1)

        # Q-Network
        self.qnetwork_local  = QNetwork(action_size, obs_len, num_features, n_atoms, linear_type, initial_sigma, factorized).to(device)
        self.qnetwork_target = QNetwork(action_size, obs_len, num_features, n_atoms, linear_type, initial_sigma, factorized).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr, weight_decay=weight_decay)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, n_multisteps, gamma, a, False)
        # Initialize time step (for updating every UPDATE_EVERY steps and TARGET_UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        #  experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.u_step = (self.u_step + 1) % self.update_every
        if self.u_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.start_since:
                experiences, target_discount, is_weights, indices = self.memory.sample(self.beta)
                new_priorities = self.learn(experiences, is_weights, target_discount)
                self.memory.update_priorities(indices, new_priorities)

        # update the target network every TARGET_UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.target_update_every
        if self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if self.qnetwork_local.training:
            switched = True
            self.qnetwork_local.eval()
        else:
            switched = False
        with torch.no_grad():
            z_probs       = F.softmax(self.qnetwork_local(state), dim=-1)
            action_values = self.supports.mul(z_probs).sum(dim=-1, keepdim=False)
        if switched:
            self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            is_weights (torch.Tensor): tensor of importance-sampling weights
            gamma (float): discount factor for the target max-Q value

        Returns
        =======
            new_priorities (List[float]): list of new priority values for the given sample
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            rows         = tuple(range(next_states.size(0)))
            a_argmax     = F.softmax(self.qnetwork_local(next_states), dim=2)\
                               .mul(self.supports)\
                               .sum(dim=2, keepdim=False)\
                               .argmax(dim=1, keepdim=False)
            p            = F.softmax(self.qnetwork_target(next_states)[rows, a_argmax], dim=1)
            tz_projected = torch.clamp(rewards + (1 - dones) * gamma * self.supports, min=self.v_min, max=self.v_max)
            # """
            b            = (tz_projected - self.v_min) / self.delta_z
            u            = b.ceil()
            l            = b.floor()
            u_updates    = b - l + u.eq(l).type(u.dtype) # fixes the problem when having b == u == l
            l_updates    = u - b
            indices_flat = torch.cat((u.long(), l.long()), dim=1)
            indices_flat = indices_flat.add(
                               torch.arange(start=0,
                                            end=b.size(0) * b.size(1),
                                            step=b.size(1),
                                            dtype=indices_flat.dtype,
                                            layout=indices_flat.layout,
                                            device=indices_flat.device).unsqueeze(1)
                           ).view(-1)
            updates_flat = torch.cat((u_updates.mul(p), l_updates.mul(p)), dim=1).view(-1)
            target_distributions = torch.zeros_like(p)
            target_distributions.view(-1).index_add_(0, indices_flat, updates_flat)
            """
            b = ((tz_projected - V_MIN) / self.delta_z).t() # transpose for later for-loop convenience
            u = b.ceil()
            l = b.floor()
            u_updates = b - l + u.eq(l).type(u.dtype)
            l_updates = u - b
            target_distributions = torch.zeros_like(p)
            for u_indices, l_indices, u_update, l_update, prob in zip(u.long(), l.long(), u_updates, l_updates, p.t()):
                target_distributions[rows, u_indices] += u_update * prob
                target_distributions[rows, l_indices] += l_update * prob
            """

        pred_distributions = self.qnetwork_local(states)
        raise Exception(actions.view(-1, 1, 1).expand(-1, -1, pred_distributions.size(2)).shape)
        pred_distributions = pred_distributions.gather(dim=1, index=actions.view(-1, 1, 1).expand(-1, -1, pred_distributions.size(2))).squeeze(1)

        """
        cross_entropy = target_distributions.mul(pred_distributions.exp().sum(dim=-1, keepdim=True).log() - pred_distributions).sum(dim=-1, keepdim=False)
        new_priorities = cross_entropy.detach().add(self.priority_eps).cpu().numpy()
        loss = cross_entropy.mul(is_weights.view(-1)).mean()
        """
        kl_divergence = F.kl_div(F.log_softmax(pred_distributions, dim=-1), target_distributions, reduce=False).sum(dim=-1, keepdim=False)
        new_priorities = kl_divergence.detach().add(self.priority_eps).cpu().numpy()
        loss = kl_divergence.mul(is_weights.view(-1)).mean()
#         """

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.clip)
        self.optimizer.step()

        return new_priorities

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def to(self, device):
        self.qnetwork_local  = self.qnetwork_local.to(device)
        self.qnetwork_target = self.qnetwork_target.to(device)
