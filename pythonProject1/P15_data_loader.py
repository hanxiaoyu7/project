import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

transform_P15 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=transform_P15, download=True)
test_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=transform_P15, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 测试数据集中第一张图片
# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("P15")
for epoch in range(2):
    step = 0  # tensorboard中的第step个步数
    for data in train_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()