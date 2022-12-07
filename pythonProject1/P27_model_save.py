import torch
import torchvision.models
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1
torch.save(vgg16, "vgg16_method1.pth") # 既能保存网络模型的结构，还能保存网络模型的参数

# 保存方式2（官方推荐，所占空间更小）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  # 把vgg16的状态保存成一种字典格式（不再保存结构，只保存参数）

# 陷阱
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = Net()
torch.save(net, "net_method1.pth")