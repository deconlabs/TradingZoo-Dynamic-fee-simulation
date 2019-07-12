import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .default_hyperparameters import SEED, N_ATOMS, INIT_SIGMA, LINEAR, FACTORIZED

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, initial_sigma=INIT_SIGMA, factorized=FACTORIZED):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.factorized = factorized
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
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
            nn.init.constant_(self.noisy_weight, self.initial_sigma / sqrt_input_size)
        else:
            bound = math.sqrt(3 / self.weight.size(1))
            nn.init.constant_(self.noisy_weight, self.initial_sigma)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            if self.factorized:
                nn.init.constant_(self.noisy_bias, self.initial_sigma / sqrt_input_size)
            else:
                nn.init.constant_(self.noisy_bias, self.initial_sigma)

    def forward(self, input):
        if self.noise:
            if self.factorized:
                input_noise  = torch.randn(1, self.noisy_weight.size(1), device=self.noisy_weight.device)
                input_noise  = input_noise.sign().mul(input_noise.abs().sqrt())
                output_noise = torch.randn(self.noisy_weight.size(0), device=self.noisy_weight.device)
                output_noise = output_noise.sign().mul(output_noise.abs().sqrt())
                weight_noise = input_noise.mul(output_noise.unsqueeze(1))
                bias_noise = output_noise
            else:
                weight_noise = torch.randn_like(self.noisy_weight)
                bias_noise = None if self.bias is None else torch.randn_like(self.noisy_bias)

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


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

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

        # Bottleneck idea from Google's MobileNetV2

        # N * obs_len * num_features
        # x.transpose(-1, -2).contiguous().unsqueeze(-1)
        # N * num_features * obs_len * 1
        self.conv0 = nn.Sequential(
            nn.LayerNorm([obs_len, 1]),
            nn.Conv2d(num_features, num_features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features * 2)
        )
        # N * (num_features*2) * 128 * 1
        self.bottleneck0 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features * 12, kernel_size=1),
            nn.BatchNorm2d(num_features * 12),
            nn.ReLU6(),
            nn.Conv2d(num_features * 12, num_features * 12, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), groups=num_features * 12),
            nn.BatchNorm2d(num_features * 12),
            nn.ReLU6(),
            nn.Conv2d(num_features * 12, num_features * 4, kernel_size=1),
            nn.BatchNorm2d(num_features * 4),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
        # N * (num_features*4) * 32 * 1
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features * 24, kernel_size=1),
            nn.BatchNorm2d(num_features * 24),
            nn.ReLU6(),
            nn.Conv2d(num_features * 24, num_features * 24, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), groups=num_features * 24),
            nn.BatchNorm2d(num_features * 24),
            nn.ReLU6(),
            nn.Conv2d(num_features * 24, num_features * 8, kernel_size=1),
            nn.BatchNorm2d(num_features * 8),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
        # N * (num_features*8) * 8 * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features * 8, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=(8, 1))
        )
        self.glb_avg_pool = nn.AvgPool2d(kernel_size=1)
        # N * 512 * 1 * 1
        # x.view(-1, 512)
        # N * 512
        self.fc_s = nn.Sequential(
            linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            linear(256, n_atoms)
        )
        self.fc_a = nn.Sequential(
            linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            linear(256, action_size * n_atoms)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state.transpose(-1, -2).unsqueeze(-1)
        x = self.conv0(x)
        x = self.bottleneck0(x)
        x = self.bottleneck1(x)
        x = self.conv1(x)
        self.glb_avg_pool.kernel_size = self.glb_avg_pool.stride = tuple(x.shape[-2:])
        x = self.glb_avg_pool(x)
        x = x.view(-1, 512)

        state_value = self.fc_s(x)

        advantage_values = self.fc_a(x)
        advantage_values = advantage_values.view(advantage_values.size()[:-1] + (self.action_size, self.n_atoms))

        dist_weights = state_value.unsqueeze(dim=-2) + advantage_values - advantage_values.mean(dim=-2, keepdim=True)

        return dist_weights

    def noise(self, enable):
        enable = bool(enable)
        for m in self.children():
            if isinstance(m, NoisyLinear):
                m.noise = enable
