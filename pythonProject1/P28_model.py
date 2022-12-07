# 搭建神经网络
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),  # (3x32x32)->(32x32x32)
            nn.MaxPool2d(kernel_size=2),  # (32x32x32)->(32x16x16)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),  # (32x16x16)->(32x16x16)
            nn.MaxPool2d(kernel_size=2),  # (32x16x16)->(32x8x8)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),  # (32x8x8)->(64x8x8)
            nn.MaxPool2d(kernel_size=2),  # (64x8x8)->(64x4x4)
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    net = Net()
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.shape)