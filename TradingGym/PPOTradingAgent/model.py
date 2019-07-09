import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTradingAgent(nn.Module):
    def __init__(self):
        super().__init__()

        # If `True`, switch the network into evaluation mode when the batch size of the input is 1.
        # This is to avoid BatchNorm error when taking a single batch of input.
        self._auto_detect_single_batch = True

        # Bottleneck idea from Google's MobileNetV2

        # N * 256 * 16
        # x.transpose(-1, -2).contiguous().unsqueeze(-1)
        # N * 16 * 256 * 1
        self.conv0 = nn.Sequential(
            nn.LayerNorm([256, 1]),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32)
        )
        # N * 32 * 128 * 1
        self.bottleneck0 = nn.Sequential(
            nn.Conv2d(32, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU6(),
            nn.Conv2d(192, 192, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), groups=192),
            nn.BatchNorm2d(192),
            nn.ReLU6(),
            nn.Conv2d(192, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
        # N * 64 * 32 * 1
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(64, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU6(),
            nn.Conv2d(384, 384, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), groups=384),
            nn.BatchNorm2d(384),
            nn.ReLU6(),
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
        # N * 128 * 8 * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=(8, 1))
        )
        # N * 512 * 1 * 1
        self.conv2 = nn.Conv2d(512, 3, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # if self._auto_detect_single_batch and self.training and x.size(0) == 1:
        #     switched = True
        #     self.eval()
        # else:
        #     switched = False

        x = x.transpose(-1, -2).unsqueeze(-1)
        x = self.conv0(x)
        x = self.bottleneck0(x)
        x = self.bottleneck1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 3)
        x = self.softmax(x)

        # if switched:
        #     self.train()

        return x