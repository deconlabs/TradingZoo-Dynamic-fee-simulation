import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTradingAgent(nn.Module):
    def __init__(self, num_features=15): #todo : num_features 하드코딩 고치기
        super().__init__()

        self.num_features = num_features

        # Bottleneck idea from Google's MobileNetV2
        self.conv0 = nn.Sequential(
            nn.LayerNorm([256, 1]),
            nn.Conv2d(num_features, num_features * 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features * 2)
        )
        # N * (num_features*2) * (n_timesteps/2) * 1
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
        # N * (num_features*4) * (n_timesteps/8) * 1
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
        # N * (num_features*8) * (n_timesteps/32) * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features * 8, 512, kernel_size=1),
            nn.BatchNorm2d(512)
        )
        self.glb_avg_pool = nn.AvgPool2d(kernel_size=1)
        # N * 512 * 1 * 1
        self.conv2 = nn.Conv2d(512, 3, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = x.transpose(-1, -2).unsqueeze(-1)
        x = self.conv0(x)
        x = self.bottleneck0(x)
        x = self.bottleneck1(x)
        x = self.conv1(x)
        self.glb_avg_pool.kernel_size = self.glb_avg_pool.stride = tuple(x.shape[-2:])
        x = self.glb_avg_pool(x)
        x = self.conv2(x)
        x = x.view(-1, 3)
        x = self.softmax(x)

        return x