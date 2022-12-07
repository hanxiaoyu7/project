import torch
import torchvision
from P27_model_save import *  # 导入所有内容

# 方式1
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # 构建网络结构，然后直接引入参数
model = torch.load("vgg16_method2.pth")


# 陷阱
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = torch.load('net_method1.pth')
print(model)
# 会报错：Can't get attribute 'Net'，说明还是需要复制网络结构，但是不需要创建实例
