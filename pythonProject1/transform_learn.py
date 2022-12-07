from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

img_path = "data/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# ToTensor
trans_totensor = transforms.ToTensor()  # 创建一个转换函数
img_tensor = trans_totensor(img)  # 进行tensor类型的转换
writer.add_image("Tensor_img", img_tensor)  # tag,img_tensor(tensor型或numpy型)

# Normalize
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  # 可以更改，效果不同
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm,1)

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)  # 由PIL变换为tensor数据
writer.add_image("Resize", img_resize, 0)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])  # 注意前面的输出与后面的输入数据类型是否相互匹配
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(200)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()
