import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("./CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_data, batch_size=16)


vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier[6] = nn.Linear(4096, 10)

print(vgg16_true)
