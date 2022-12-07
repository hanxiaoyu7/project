import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


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


net = Net()
print(net)

writer = SummaryWriter("log")
step = 0
for data in dataloader:  # 从dataloader取数据的方法
    imgs, targets =  data
    #writer.add_images("input", imgs, step)
    print(imgs.shape)
    output = net(imgs)  # 有没有一次性把图片集输入神经网络的方法？？
    #writer.add_images("output_linear", output, step)  # 注意，是图片的输出
    print(output.shape)
    step = step + 1

writer.close()
