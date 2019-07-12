import math
import numpy as np
import torch
from collections import deque, namedtuple

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, n_agents, buffer_size, batch_size, n_multisteps, gamma, a, separate_experiences):
        """Initialize a ReplayBuffer object.

        Params
        ======
            n_agents (int): number of agents, or simulations, running simultaneously
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            n_multisteps (int): number of time steps to consider for each experience
            gamma (float): discount factor
            a (float): priority exponent parameter
            separate_experiences (bool): whether to store experiences with no overlap
        """
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_multisteps = n_multisteps
        self.gamma = gamma
        self.a = a
        self.separate_experiences = bool(separate_experiences)

        self.memory_write_idx = 0
        self._non_leaf_depth = math.ceil(math.log2(buffer_size))
        self._memory_start_idx = 2 ** self._non_leaf_depth
        self._buffer_is_full = False
        self.memory = [None for _ in range(buffer_size)]
        self.priorities_a = np.zeros(buffer_size)
        self.tree = np.zeros(self._memory_start_idx + buffer_size) # starts from index 1, not 0; makes implementation easier and reduces many small computations

        self.multistep_collectors = [deque(maxlen=n_multisteps) for _ in range(n_agents)]
        self.max_priority_a = 1.
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self._divisors = np.power(2, np.arange(1, self._non_leaf_depth + 1)).reshape((-1, 1))
        self._discounts = np.power(self.gamma, np.arange(self.n_multisteps + 1))
        self._target_discount = float(self._discounts[-1])

    def add(self, i_agent, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        assert isinstance(i_agent, int)
        assert 0 <= i_agent < self.n_agents
        collector = self.multistep_collectors[i_agent]
        e = self.experience(state, action, reward, next_state, done)
        collector.append(e)
        if len(collector) == self.n_multisteps:
            self.memory[self.memory_write_idx] = tuple(collector)

            delta_priority_a = self.max_priority_a - self.priorities_a[self.memory_write_idx]
            tree_idx = self._memory_start_idx + self.memory_write_idx
            self.priorities_a[self.memory_write_idx] = self.max_priority_a
            self.tree[tree_idx] = self.max_priority_a
            # tree_indices = np.floor_divide(tree_idx, self._divisors).reshape((-1,))
            # np.add.at(self.tree, tree_indices, np.tile(delta_priority_a, self._non_leaf_depth))
            for _ in range(self._non_leaf_depth):
                tree_idx = tree_idx // 2
                self.tree[tree_idx] += delta_priority_a

            self.memory_write_idx += 1
            if self.memory_write_idx >= self.buffer_size:
                self._buffer_is_full = True
                self.memory_write_idx = 0

            if self.separate_experiences:
                collector.clear()
        if done:
            collector.clear()

    def sample(self, beta):
        """Randomly sample a batch of experiences from memory.

        Params
        ======
            beta (int or float): parameter used for calculating importance-priority weights

        Returns
        =======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            target_discount (float): discount factor for target max-Q value
            is_weights (torch.Tensor): tensor of importance-sampling weights
            indices (np.ndarray): sample indices"""
        indices_not_prepared = True
        sample_value_basis = np.linspace(0, self.tree[1], num=self.batch_size, endpoint=False, dtype=np.float32)
        while indices_not_prepared:
            sample_values = np.add(sample_value_basis, np.multiply(np.random.rand(self.batch_size), sample_value_basis[1]))
            tree_indices = np.ones(self.batch_size, dtype=np.int32)
            try:
                for _ in range(self._non_leaf_depth):
                    left_child_indices = np.multiply(tree_indices, 2)
                    right_child_indices = np.add(left_child_indices, 1)
                    greater_than_left = np.greater(sample_values, self.tree[left_child_indices])
                    sample_values = np.where(greater_than_left, np.subtract(sample_values, self.tree[left_child_indices]), sample_values)
                    tree_indices = np.where(greater_than_left, right_child_indices, left_child_indices)
            except IndexError: # Don't know exactly why it occurs. Suspecting numerical error issues with floating numbers as a probable cause
                continue
            else:
                indices_not_prepared = False
        indices = np.subtract(tree_indices, self._memory_start_idx)

        experiences = tuple(zip(*[self.memory[i] for i in indices if self.memory[i] is not None]))

        first_states = torch.tensor([e[0] for e in experiences[0]], dtype=torch.float, device=device)
        actions      = torch.tensor([e[1] for e in experiences[0]], dtype=torch.float, device=device)
        rewards      = torch.tensor(
                           np.sum(
                               np.multiply(
                                   np.array([[e[2] for e in experiences_step] for experiences_step in experiences]).transpose(), self._discounts[:-1]
                               ), axis=1, keepdims=True
                           ), dtype=torch.float, device=device)
        last_states  = torch.tensor([e[3] for e in experiences[-1]], dtype=torch.float, device=device)
        dones        = torch.tensor([e[4] for e in experiences[-1]], dtype=torch.float, device=device).view(-1, 1)

        is_weights = np.divide(self.priorities_a[indices], self.tree[1])
        is_weights = np.power(np.multiply(is_weights, self.buffer_size if self._buffer_is_full else self.memory_write_idx), -beta)
        is_weights = torch.tensor(np.divide(is_weights, np.max(is_weights)).reshape((-1, 1)), dtype=torch.float, device=device)

        return (first_states, actions, rewards, last_states, dones), self._target_discount, is_weights, indices

    def update_priorities(self, indices, new_priorities):
        """Update the priorities for the experiences of given indices to the given new values.

        Params
        ======
            indices (array_like): indices of experience priorities to update
            new_priorities (array-like): new priority values for given indices"""
        # Remove Duplicate Samples (discard except the first occurence in the array)
        indices, idx_indices = np.unique(indices, return_index=True)
        new_priorities_a = np.power(new_priorities[idx_indices], self.a)

        delta_priority_a = np.subtract(new_priorities_a, self.priorities_a[indices])
        tree_indices = np.add(indices, self._memory_start_idx)
        self.priorities_a[indices] = new_priorities_a
        self.tree[tree_indices] = new_priorities_a
        tree_indices = np.floor_divide(tree_indices, self._divisors).reshape((-1,))
        np.add.at(self.tree, tree_indices, np.tile(delta_priority_a, self._non_leaf_depth))

        self.max_priority_a = np.max(self.priorities_a)

    def reset_multisteps(self, i_agent=-1):
        assert isinstance(i_agent, int) and -1 <= i_agent < self.n_agents
        if i_agent == -1:
            for collector in self.multistep_collectors:
                collector.clear()
        else:
            self.multistep_collectors[i_agent].clear()

    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_size if self._buffer_is_full else self.memory_write_idx
