import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .default_hyperparameters import SEED, N_ATOMS, INIT_SIGMA, LINEAR, FACTORIZED


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.factorized = factorized
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('noisy_bias', None)
        self.reset_parameters()

        self.noise = True

    def reset_parameters(self):
        if self.factorized:
            sqrt_input_size = math.sqrt(self.weight.size(1))
            bound = 1 / sqrt_input_size
            nn.init.constant_(self.noisy_weight,
                              self.initial_sigma / sqrt_input_size)
        else:
            bound = math.sqrt(3 / self.weight.size(1))
            nn.init.constant_(self.noisy_weight, self.initial_sigma)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            if self.factorized:
                nn.init.constant_(
                    self.noisy_bias, self.initial_sigma / sqrt_input_size)
            else:
                nn.init.constant_(self.noisy_bias, self.initial_sigma)

    def forward(self, input):
        if self.noise:
            if self.factorized:
                input_noise = torch.randn(1, self.noisy_weight.size(
                    1), device=self.noisy_weight.device)
                input_noise = input_noise.sign().mul(input_noise.abs().sqrt())
                output_noise = torch.randn(self.noisy_weight.size(
                    0), device=self.noisy_weight.device)
                output_noise = output_noise.sign().mul(output_noise.abs().sqrt())
                weight_noise = input_noise.mul(output_noise.unsqueeze(1))
                bias_noise = output_noise
            else:
                weight_noise = torch.randn_like(self.noisy_weight)
                bias_noise = None if self.bias is None else torch.randn_like(
                    self.noisy_bias)

            if self.bias is None:
                return F.linear(
                    input,
                    self.weight.add(self.noisy_weight.mul(weight_noise)),
                    None
                )
            else:
                return F.linear(
                    input,
                    self.weight.add(self.noisy_weight.mul(weight_noise)),
                    self.bias.add(self.noisy_bias.mul(bias_noise))
                )
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, initial_sigma={}, factorized={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.initial_sigma, self.factorized
        )


class MultiHeadedAttention(nn.Module):
    """Most part of the implementation from http://nlp.seas.harvard.edu/2018/04/03/attention"""

    def __init__(self, h, d_model):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.attn = None

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.query_linear(query).view(
            nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(
            nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(
            nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.output_linear(x)

    @staticmethod
    def attention(query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    class AttentionFFLayer(nn.Module):
        def __init__(self, in_out_size, h=8):
            assert in_out_size % h == 0, "The input size must be divisible by the number of attention heads `h`."
            super().__init__()
            self.in_out_size = in_out_size
            self.h = h

            self.norm0 = nn.LayerNorm(in_out_size)
            self.attn = MultiHeadedAttention(h, in_out_size)
            self.norm1 = nn.LayerNorm(in_out_size)
            self.fc = nn.Sequential(
                nn.Linear(in_out_size, in_out_size * 4),
                nn.ReLU(),
                nn.Linear(in_out_size * 4, in_out_size)
            )

        def forward(self, x):
            # same input for `query`, `key`, and `value`
            x = x + self.attn(*([self.norm0(x)]*3))
            x = x + self.fc(self.norm1(x))
            return x

    class FinalFFAttentionLayer(nn.Module):
        def __init__(self, in_out_size, h=8):
            assert in_out_size % h == 0, "The input size must be divisible by the number of attention heads `h`."
            super().__init__()
            self.in_out_size = in_out_size
            self.h = h

            self.norm0 = nn.LayerNorm(in_out_size)
            self.fc = nn.Linear(in_out_size, in_out_size)
            self.norm1 = nn.LayerNorm(in_out_size)
            self.attn = MultiHeadedAttention(h, in_out_size)
            self.norm2 = nn.LayerNorm(in_out_size)

        def forward(self, memory):
            assert memory.dim() == 3
            x = memory[:, -1:]
            x = x + self.fc(self.norm0(x))
            x = x + self.attn(self.norm1(x), memory, memory)
            return self.norm2(x).squeeze(1)

    class PositionalEncoding(nn.Module):
        "Implement the PE function."

        def __init__(self, d_model, dropout, max_len=5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0., max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2) *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)],
                             requires_grad=False)
            return self.dropout(x)

    def __init__(self, action_size, obs_len, num_features=16, n_atoms=N_ATOMS, linear_type=LINEAR, initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
            num_features (int): Number of features in the state
            n_atoms (int): number of support atoms
            linear_type (str): type of linear layers ('linear', 'noisy')
            initial_sigma (float): initial weight value for noise parameters
                when using noisy linear layers
        """
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.obs_len = obs_len
        self.num_features = num_features
        self.n_atoms = n_atoms
        self.linear_type = linear_type.lower()
        self.factorized = bool(factorized)

        def noisy_layer(in_features, out_features):
            return NoisyLinear(in_features, out_features, True, initial_sigma, factorized)
        linear = {'linear': nn.Linear, 'noisy': noisy_layer}[self.linear_type]

        # N * obs_len * num_features
        # x.view(-1, -1, x.size(-1))
        # (N * obs_len) * num_features
        self.fc0 = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 64)
        )
        # x.view(N, obs_len, x.size(-1))
        # N * obs_len * 64
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        # N * obs_len * 128
        self.inner_layers = nn.Sequential(
            self.AttentionFFLayer(64),
            self.AttentionFFLayer(64),
            self.AttentionFFLayer(64),
            self.FinalFFAttentionLayer(64)
        )
        # N * 64
        self.fc_s = nn.Sequential(
            linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            linear(32, n_atoms)
        )
        self.fc_a = nn.Sequential(
            linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            linear(32, action_size * n_atoms)
        )
        self.pe = self.PositionalEncoding(64,.1)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.fc0(x.view(-1, x.size(-1))).view(x.size(0), x.size(1), -1)
        x = self.pe(x)

        x = self.inner_layers(x) # FF, FF, FF, finalFF

        state_value = self.fc_s(x) # double-dqn : state

        advantage_values = self.fc_a(x) # double-dqn : advantage
        advantage_values = advantage_values.view(
            advantage_values.size()[:-1] + (self.action_size, self.n_atoms))

        dist_weights = state_value.unsqueeze(
            dim=-2) + advantage_values - advantage_values.mean(dim=-2, keepdim=True)

        return dist_weights

    def noise(self, enable):
        enable = bool(enable)
        for m in self.children():
            if isinstance(m, NoisyLinear):
                m.noise = enable
