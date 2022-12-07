from torch.utils.data import Dataset
from PIL import Image
import os # 一个库，可以获取所有图片的地址


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):  # 初始化函数，为class提供一个全局变量，为后面的函数提供它们所需要的量
        self.root_dir = root_dir  # train文件夹地址
        self.label_dir = label_dir  # ants
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 获得所有ants图片名称列表 # self.：指定了类当中的全局变量，后面的函数也可以使用，相当于java里的this

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取具体某张图片的名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 获取所有ants图片地址列表
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path) # 返回列表长度


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)


train_dataset = ants_dataset + bees_dataset # 数据集的拼接