import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True) # 建议target=True，因为反正如果下载过了也不会重新下载
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
# 提高下载速度的方法：运行文件，复制控制台的下载链接，在迅雷中进行下载

print(test_set)

# 在tensorboard里显示导入的数据集图片
writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_Set", img, i)

writer.close()