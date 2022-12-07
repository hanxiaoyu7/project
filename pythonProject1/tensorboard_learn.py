import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter这个类

writer = SummaryWriter("logs")  # 给类创建一个实例
image_path = "data/train/bees_image/39747887_42df2855ee.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)


writer.add_image("train", img_array, 1, dataformats="HWC")  # 会使用到的两个方法

writer.close()
