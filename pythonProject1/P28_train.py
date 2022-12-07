import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P28_model import *

train_data = torchvision.datasets.CIFAR10("./CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用dataloader加载
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络实例
net = Net()
loss_fun = nn.CrossEntropyLoss()
learning_rate = 0.01
optim = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置网络参数
total_train_step = 0  # 记录训练次数，一个batch为一次训练
total_test_step = 0  # 记录测试次数
epoch = 10  # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("./log")

for i in range(epoch):
    print("-------------第{}轮训练开始------------".format(i+1))
    # 训练开始
    net.train()  # 把网络设置成训练格式，只对特定的层（如Dropout层）有作用
    for data in train_dataloader:
        imgs, labels = data
        output = net(imgs)  # 注意是把imgs传入不是把data传入

        # 优化器优化模型
        loss = loss_fun(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 200 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))  # loss和loss.item()表面没有区别，实际后者把tensor转化成了正常python值
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    net.eval()  # 把网络设置成测试格式，只对特定的层（如Dropout层）有作用
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 取消梯度：在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，反向传播时就不会自动求导了，因此大大节约了显存或者说内存
        for data in test_dataloader:
            imgs, labels = data
            output = net(imgs)
            loss = loss_fun(output, labels)
            total_test_loss = total_test_loss + loss.item()
            accuracy = ((output.argmax(1) == labels).sum())
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存训练运行数据
    torch.save(net, "net_{}.pth".format(i))  # 保存每一轮训练结果，其中i为第i个epoch
    print("模型已保存")

writer.close()