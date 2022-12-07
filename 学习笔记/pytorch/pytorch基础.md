# pytorch基础教程

> [参考教程](https://bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=28f4ef64be559a1c9c4ab1cf2b8ee826)

## 四个包

PyTorch由4个主要包装组成：
1.Torch：类似于Numpy的通用数组库，可以在将张量类型转换为（torch.cuda.TensorFloat）并在GPU上进行计算。
2.torch.autograd：用于构建计算图形并自动获取渐变的包
3.torch.nn：具有共同层和成本函数的神经网络库
4.torch.optim：具有通用优化算法（如SGD，Adam等）的优化包

## 两大重要函数

- `dir()`：打开
  - `dir(pytorch)`-->输出1、2、3、4分割栏
  - `dir(python.3)`-->输出a,b,c工具
  
- `help()`：说明
  - `help(pytorch.3.a)`-->输出：a工具的用途

- 应用实例

  ```python
  dir(torch)
  dir(torch.cuda)
  dir(torch.cuda.is_available())
  help(torch.cuda.is_available) # 注意这里要去掉括号
  ```

## Tensor张量

> torch.Tensor是存储和变换数据的主要工具，是 Pytorch 最主要的库
>
> Pytorch 的一大作用就是可以代替 Numpy 库，所以首先介绍 Tensors ，也就是张量，它相当于 Numpy 的多维数组(ndarrays)。两者的区别就是 Tensors 可以应用到 GPU 上加快计算速度。
>
> 可以理解为tensor中包含了神经网络所需要的参数
>
> [参考教程](https://blog.csdn.net/qq_43328040/article/details/107420004?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166789778216800182115141%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166789778216800182115141&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-107420004-null-null.142^v63^control_1,201^v3^add_ask,213^v1^t3_esquery_v3&utm_term=tensor&spm=1018.2226.3001.4187)（翻译自官网文档）

### 1. 创建Tensor

- `x = torch.empty(5,3)`
- `x = torch.rand(5,3)`或`randn(5,3)`或`randi([a,b],5,3)`
  - `rand`生成0~1间均匀分布的伪随机数
  - `randn`生成标准正态分布的伪随机数（均值为0，方差为1）
  - `randi`生成a~b间均匀分布的伪随机整数，其中a省略时默认为0
- `x = torch.zeros(5,3,dtype=torch.long)`：创建5x3的long型全0的Tensor
- `x = torch.tensor([6,7,8])`：直接根据数据创建Tensor
- `x = arange(a,b,step)`：从a到b，步长为step，若未给step则默认为1
- `x = linspace(a,b,n)`：从a到b，均匀切分为n份

- 更多操作详见官网，比较类似matlab生成矩阵

### 2. 操作Tensor

#### ① 算术运算操作

- 加法操作：
  - `x+y`
  - 或`torch.add(x,y,out=result)`
  - 或`y.add_(x)`（相当于y+=x，注意后缀`_`）

#### ② 矩阵运算操作

- `x.t()`：矩阵转置

- `x.diag()`：提取x的对角线元素

- `x.inverse()`：求x的逆矩阵

- `x+y`：矩阵相加；若两个矩阵尺寸不同，则会触发**广播机制**（一种复制机制）

  ```python
  x = arange(1,3) # x = [1,2]
  y = arange(1,4).view(3,1) # y = [1;2;3]
  x + y
  # x+y=[2,3;3,4;4,5]
  ```

- 注意：矩阵运算操作`y = x + y`会**开辟新内存**！即前后`y`的地址产生了变化！

#### ③ 索引

- 类同matlab，但注意：索引出的结果与原数据**共享内存**！即修改一个另一个也会跟着修改

  ```python
  y = x[0,:]
  y += 1
  print(y)
  print(x[0, :]) # 会发现结果一样
  ```

- 还有一点不同之处在于`y=x[1:3]`实际索引的是前两行，不是三行

#### ④ 查询尺寸：x.shape()或x.size()

- `x.shape()`或`x.size()`：获取tensor的尺寸
- 返回的torch.size数据类型为tuple（元组，类似列表，一个坑），支持所有tuple的操作

#### ⑤ 改变尺寸：x.view()

- `y = x.view(15)`：把5x3的`x`转化为1x15的`y`

- `z = x.view(-1,5)`：-1所指维度可以根据其他的维度推导得出

- 注意：新tensor与源tensor的尺寸不同，但仍**共享data**！修改一个另一个也会改变！（但是id不同，仍然是两个不同的Tensor）

- 若想不共享数据内存，则需要先`x.clone()`再view：

  ```python
  x = torch.rand(5,3)
  x2= x.clone()
  y = x2.view(15)
  y += 1
  print(x)
  print(y)
  ```

#### ⑥ 改变维度：squeeze与unsqueeze

- `y = torch.squeeze(x,i)`：降维，去除张量中第i个地方的1维度；若不给i，则去除所有的1维度

- `y = x.unsqueeze(i)`：升维，在第i个地方加一个1维度

  - 升维前：

    ```python
    import torch
    input=torch.arange(0,6)
    print(input)
    print(input.shape)
    结果：
    tensor([0, 1, 2, 3, 4, 5])
    torch.Size([6])
    ```

  - 在0处升维后：

    ```python
    print(input.unsqueeze(0))
    print(input.unsqueeze(0).shape)
    结果：
    tensor([[0, 1, 2, 3, 4, 5]])
    torch.Size([1, 6])
    ```

  - 在1处升维后：

    ```python
    print(input.unsqueeze(1))
    print(input.unsqueeze(1).shape)
    结果：
    tensor([[0],
            [1],
            [2],
            [3],
            [4],
            [5]])
    torch.Size([6, 1])
    ```

    

#### ⑦ 改变数据类型

- **Tensor→普通python number：**`y = x.item()`

- **Tensor→NumPy：**`y = x.numpy`【此方法会共享内存！】

- **xxx→Numpy：**`y = np.array(x)`（需要导入，详见下方Tensorboard的使用部分）

- **NumPy→Tensor：**

  - `y = torch.from_numpy(x)`【此方法会共享内存！】

  ```python
  import numpy as np
  import torch
  a = np.ones(5)
  b = torch.from_numpy(a)
  print(a, b)
  
  a += 1
  print(a, b) # 发现a和b都加了
  b += 1
  print(a, b) # 发现a和b都加了
  ```

  - `y = torch.tensor(x)`：此方法不会共享内存，会进行数据拷贝

- **xxx→Tensor：**利用ToTensor（需要导入，详见下方Transform的使用部分）

#### ⑧ 查找内存位置：id(x)

- 验证两个实例是否占用同一内存：

  ```python
  x = torch.tensor([1,2])
  y = torch.tensor([3,4])
  id_before = id(y)
  y = y + x
  print(id(y) == id_before) # 结果为False
  ```

- 若想相加操作不开辟新内存，则如下操作：

  ```python
  x = torch.tensor([1,2])
  y = torch.tensor([3,4])
  id_before = id(y)
  y[:] = y + x
  # 或y += x
  # 或y.add_(x)
  # 或torch.add(x,y,out=y)
  print(id(y) == id_before) # 结果为True
  ```

- 注：虽然`view`返回的`Tensor`与源`Tensor`是共享`data`的，但是依然是一个新的`Tensor`（因为`Tensor`除了包含`data`外还有一些其他属性），二者id（内存地址）并不一致

### 3. 在CPU与GPU间转换

暂时用不到，略，见教程

## torchvision介绍

- torchvision官网文档：[pytorch.torchvision](https://pytorch.org/vision/stable/index.html)
- torchvision主要包含以下模块
  - dataset：数据集下载
  - transforms：提供图片变换
  - tensorboard：提供常用小工具
  - models：提供训练好的神经网络
  - torchvision的io模块、ops模块不常用，不作介绍

### torchvision自带数据集的使用

> 以CIFAR10数据集为例

- 用法：`dataset = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)`
  - 注：不知道下一个参数该写什么时，可以ctrl+P

- 实例：

  ```python
  import torchvision
  from torch.utils.tensorboard import SummaryWriter
  
  dataset_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
  ])
  train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True) # 建议target=True，因为反正如果下载过了也不会重新下载
  test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
  # 提高下载速度的方法：运行文件，复制控制台的下载链接，在迅雷中进行下载
  
  print(test_set[0])  # 查询内容用[]
  print(test_set.classes)  # 查询属性用.
  
  img, target = test_set[0]
  print(img)
  print(target)
  print(test_set.classes[target])
  img.show()
  
  print(test_set[0])  # 输出的是tensor数据
  
  # 在tensorboard里显示导入的数据集图片
  writer = SummaryWriter("p10")
  for i in range(10):
      img, target = test_set[i]
      writer.add_image("test_Set", img, i)
  
  writer.close()
  ```

  - 提高数据集下载速度的方法：运行代码，复制控制台的下载链接，在迅雷中进行下载；如果运行里没有出链接，则ctrl+CIFAR10点开往上划，也能看到链接
  - test_set下**有一个属性是classes**，是一个数组，包含了该数据集所居有的十个类别（cat,dog,plane等）

## Tensorboard的使用

> Tensorboard是一个可视化工具

### Tensorboard的安装与打开

- tensorboard安装：在终端中输入`pip install tensorboard`

- 返回tensorboard打开的窗口：在终端中输入`tensorboard --logdir=...(对应的日志文件夹名)` 

- 如果一个主机上有好多人使用，都打开同一窗口会出问题，可指定端口号

  `tensorboard --logdir=logs --port=6007`

- 使用时，先run文件，再打开端口网址，即可看到图像

- 注：若运行文件出现错误`AttributeError: module ‘distutils‘ has no attribute ‘version‘`，pip或者conda install setuptools==58.0.4

- 注：每次重新运行需要删除log

### 主文件书写

- **框架**：导入+创建实例+两个函数+关闭

#### 1. 导入SummaryWriter类

- `from torch.utils.tensorboard import SummaryWriter`：导入所需要的类

- ctrl+点击SummaryWriter可以看怎么写相应__init\_\_

#### 2. 创建实例

- `writer = SummaryWriter("logs")`

#### 3. 常用函数

- **绘制函数图像**：`writer.add_scalar(tag, y, x)`

  - tag：图像的title

  - y：y轴

  - x：x轴，即训练的步数

  - 例子

    ```
    for i in range(100)
    	writer.add_scalar("y=2x",2i,i)
    ```

  - 删掉log下的所有文件

- **显示图片**：`writer.add_image(tag, img_tensor, global_step=None)`

  - tag：图像的title

  - img_tensor：**需要是tensor或numpy型数据**，注意需要转换！

    - ```python
      # 观察当前图片的数据类型
      img = Image.open(image_path)
      type(img)
      ```

    - ```python
      # 把PIL格式图片转化为numpy型数据
      import numpy as np
      img_array = np.array(img)
      # 也可以利用Transform中的ToTensor，把PIL格式图片转化为tensor型数据
      from torchvision import transforms
      tensor_trans = transforms.ToTensor()  # 创建一个转换函数
      tensor_img = tensor_trans(img)  # 进行tensor类型的转换
      ```

  - global_step（int）：全局步值
  
- 生成神经网络流程图：`writer.add_graph(net_name, input)`

  - net_name：神经网络实例
  - input：输入数据
  - 不常用，但可用于研究神经网络


#### 4. 关闭

- `writer.close()`

### 用途

- 神经网络训练到一定step时，绘制loss函数的图像

### 实例

- add_scalar实例

```python
from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter这个类

writer = SummaryWriter("logs") # 给类创建一个实例

# y = x
for i in range(100):
    writer.add_scalar("y=x", i, i) # title,y_label,x_label

writer.close()

```

执行代码后打开上述网址，显示如下：

![](E:\学习笔记\pic\pytorch基础2.png)

- add_image实例

  ```python
  import numpy as np
  from PIL import Image
  from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter这个类
  
  writer = SummaryWriter("logs")  # 给类创建一个实例
  image_path = "data/train/bees_image/39747887_42df2855ee.jpg"
  img_PIL = Image.open(image_path)
  img_array = np.array(img_PIL)
  # print(type(img_array))  >>> numpy
  # print(img_array.shape)  >>> (250, 250, 3)，说明是HWC
  
  
  writer.add_image("train", img_array, 1, dataformats="HWC") 
  
  writer.close()
  
  ```

  可以换张图片，全局步长换为2，再运行一次，刷新tensorboard，发现可以拖动换图

## Transform的使用

> 指transforms.py工具箱，包含totensor、resize等常见工具，用于处理图片格式

- `from torchvision import transforms`：导入需要的库

- 对transform的理解

  1. 先创建具体的工具（必要时传入参数）
  2. 再使用具体的工具（传入自变量）

  ![](E:\学习笔记\pic\pytorch6.png)



- 注意事项
  - 需要格外注意输入与输出数据的数据类型（PIL/numpy/Tensor？），可以通过print(type(变量))查询数据类型
  - 关注官方文档

### 常见的Transform

> 可以直接ctrl+Tranform库，查看所有Transform

- `ToTensor()`：

  - 功能：**将PIL或numpy数据类型转化为tensor**

  - 用法：

    ```python
    trans_tensor = transforms.ToTensor()  # 创建一个转换函数
    tensor_img = trans_tensor(img)  # 进行tensor类型的转换
    ```

- `Normalize(mean, std, inplace=False)`

  - 功能：**归一化**

    - $$
      input[channel]=(input[channel]-mean[channel])/std[channel]
      $$

    - 若mean、std=0.5，输入范围[0,1]，则归一化后输出范围[-1,1]

  - 用法：输入输出可以是PIL型

    ```python
    trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    img_norm = trans_norm(img_tensor)
    print(img_norm[0][0][0])
    writer.add_image("Normalize", img_norm)
    ```

- `Resize(size)`

  - 功能：**重置图片尺寸**

  - 用法：输入输出可以是PIL型

    ```python
    print(img.size)  # 原始尺寸
    trans_resize = transforms.Resize((512, 512))
    img_resize = trans_resize(img)
    print(img_resize)  # 新的尺寸
    ```

- `Compose([transforms参数1, transforms参数2, ...])`

  - 功能：**串联多个图片变换**

  - 用法：Compose中的参数需要是一个列表：[数据1，数据2，...]，在Compose中，数据需要是transforms类型

    ```python
    trans_resize_2 = transforms.Resize(512)
    trans_compose = transforms.Compose([trans_resize_2, trans_totensor])  # 注意前面的输出与后面的输入数据类型是否相互匹配
    img_resize_2 = trans_compose(img)
    writer.add_image("Resize", img_resize_2, 1)
    ```

  - 注：Compose的各个参数间，前面一个的输出需要与后面一个的输入的数据格式匹配

- `RandomCrop(size)`

  - 功能：**随机裁剪**（注意Resize是等比缩放）

  - 用法：输入输出可以是PIL型

    ```python
    trans_random = transforms.RandomCrop(200)
    trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
    for i in range(10):
        img_crop = trans_compose_2(img)
        writer.add_image("RandomCropHW", img_crop, i)
    ```

  - 使用场景：神经网络——数据增强


### Transform在导入torchvision数据集时的使用实例

- transform更多的使用不是单独使用，而是作为函数嵌在数据集的导入中

- （实例见torchvision部分）

## autograd库

> pytorch的自动求导库，PyTorch中所有的神经网络都来自于autograd包
>
> 现代深度学习系统中（比如MXNet， TensorFlow等）都用到了一种技术——**自动微分**。在此之前，机器学习社区中很少发挥这个利器，一般都是用Backpropagation进行梯度求解，然后进行SGD等进行优化更新。手动实现过backprop算法的同学应该可以体会到其中的复杂性和易错性，一个好的框架应该可以很好地将这部分难点隐藏于用户视角，而自动微分技术恰好可以优雅解决这个问题。
>
> 对于 Pytorch 的神经网络来说，非常关键的一个库就是 `autograd` ，它主要是提供了对 Tensors 上所有运算操作的自动微分功能，也就是计算梯度的功能。它属于 `define-by-run` 类型框架，即反向传播操作的定义是根据代码的运行方式，因此每次迭代都可以是不同的。

没看懂，以后再说

## 读取数据

> Pytorch的数据读取主要有两个重要的类：Dataset、DataLoader
>
> 参考教程：[1.一篇很详细的源码解读](https://blog.csdn.net/wuzhongqiang/article/details/105499476)、[2](https://zhuanlan.zhihu.com/p/30934236#:~:text=Pytorch%E7%9A%84%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E4%B8%BB%E8%A6%81%E5%8C%85%E5%90%AB%E4%B8%89%E4%B8%AA%E7%B1%BB%3A%20Dataset%3B%20DataLoader%3B%20DataLoaderIter%3B%20%E8%BF%99%E4%B8%89%E8%80%85%E5%A4%A7%E8%87%B4%E6%98%AF%E4%B8%80%E4%B8%AA%E4%BE%9D%E6%AC%A1%E5%B0%81%E8%A3%85%E7%9A%84%E5%85%B3%E7%B3%BB%3A%201.%E8%A2%AB%E8%A3%85%E8%BF%9B2.%2C%202.%E8%A2%AB%E8%A3%85%E8%BF%9B3.%20%E4%B8%80.,torch.utils.data.Dataset.%20%E6%98%AF%E4%B8%80%E4%B8%AA%E6%8A%BD%E8%B1%A1%E7%B1%BB%2C%20%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84Dataset%E9%9C%80%E8%A6%81%E7%BB%A7%E6%89%BF%E5%AE%83%E5%B9%B6%E4%B8%94%E5%AE%9E%E7%8E%B0%E4%B8%A4%E4%B8%AA%E6%88%90%E5%91%98%E6%96%B9%E6%B3%95%3A%20__getitem__%28%29%20__len__%28%29%20%E7%AC%AC%E4%B8%80%E4%B8%AA%E6%9C%80%E4%B8%BA%E9%87%8D%E8%A6%81%2C%20%E5%8D%B3%E6%AF%8F%E6%AC%A1%E6%80%8E%E4%B9%88%E8%AF%BB%E6%95%B0%E6%8D%AE.%20%E4%BB%A5%E5%9B%BE%E7%89%87%E4%B8%BA%E4%BE%8B%3A)、[3.transform数据预处理大全](https://blog.csdn.net/u011995719/article/details/85107009)

### 数据在文件中的组织形式

**方式1**

- train
  - ants
  - bees
- val
  - ants
  - bees

**方式2**

- train_images
- train_labels（对应的n个.txt文件）
- val_images
- val_labels

**方式3**

- 直接把label作为image的命名

### 读取图片的两种方式

#### 1. 获取PIL格式：使用Image.open()

```python
from PIL import Image
PIL_img = Image.open(img_path)  # PIL_img中包含了图片的各种属性
type(PIL_img)
>>> PIL
```

#### 2. 获取numpy格式：使用openCV

```python
# 首先在终端安装opencv：pip install opencv-python
import cv2
cv_img = cv2.imread(img_path)  # cv_img中包含了图片的各种属性
type(cv_img)
>>> numpy
```

### 获取图片地址

在左侧资源管理器copy path或copy relative path

`img_path = "C:\\\\..."`str型

`img = Image.open`可以获取img的各种属性

`img.size`：获取该图片的尺寸

`img.show()`：展示这张图片

`root_dir = "data/train"`

`label_dir = "ants"`

`path = ps.path.join(root_dir, label_dir) `

### **Dataset**：获取每个数据及其label

> torchvision里的数据集不需要自己再写Dataset，可以直接获取，见torchvision部分

- 所有自定义的Dataset都需要继承`torch.utils.data.Dataset`类，并且必须复写`__getitem__()`、`__len__()`这两个方法

#### 1. 导入所需的包

```python
# 导入所需的包
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
```

#### 2. 创建自定义Dataset类，并__init\_\_

```python
# 创建自定义Dataset类
class RMBDataset(Dataset): # 注意要继承Dataset
    def __init__(self, data_dir, transform=None):
            """
            rmb面额分类任务的Dataset
            :param data_dir: str, 数据集所在路径
            :param transform: torch.transform，数据预处理
            """
            super(RMBDataset, self).__init__()
            self.label_name = {"1": 0, "100": 1}
            self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
            self.transform = transform
            
# 或
# 注：数据储存形式为方式1
class AnimalDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir # 根路径
        self.label_dir = label_dir # 不同的label（ants或bees）
        self.path = os.path.join(self.root_dir, self.label_dir) # 合成路径
        self.img_path = os.listdir(self.path) # 把路径下的每张图片生成list格式，list内容为图片名称
```

- 如果我们想要提高模型的泛化能力，就得使用transform，对图片进行数据中心化，缩放，裁剪，填充等的一些操作

#### 3. __getitem\_\_

- `__getitem__`方法的是Dataset的核心，作用是接收一个索引，返回当前位置的数据和标签

  ```python
      # 下面为一个例子
  def __getitem__(self, index):
          path_img, label = self.data_info[index]
          img = Image.open(path_img).convert('RGB')     # 0~255，不懂这一步为什么要convert
  
          if self.transform is not None:
              img = self.transform(img)   # 在这里做transform，转为tensor等等
  
          return img, label
      
  # 或
  def __getitem__(self,index):
      img_name = self.img_path[index] # 获取第index张图片的名称
      img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 获取第Index张图片的地址
      img = Image.open(img_item_path) # 获取第index张图片的各种属性
      label = self.label_dir # 获取第index张图片的label
      return img, label
  ```

#### 4. __len\_\_

- `__len__()`方法的作用是：返回整个数据集的长度

  ```python
  def __len__(self):
      return len(self.data_info)
      
  # 或
  def __len__(self):
      return len(self.img_path) # 获取数据list的长度
  ```

#### 5. 创建类的实例

```python
# 创建自定义Dataset类的实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 或
root_dir = ...
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = AnimalDataset(root_dir, ants_label_dir)
bees_dataset = AnimalDataset(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
```

#### 6. 在控制台进行测试

```python
ants_dataset[0] # 即调用__getitem__，这是python类中的一种特殊方法
>>> (<图片信息>,'ants') 
img, label = ants_dataset[0]
img.show
>>> (弹出对应图片)
len(ants_dataset)  # 调用__len__，查询数据集个数
>>> 238
```

### DataLoader

- `torch.utils.data.DataLoader`是数据读取的重要接口

- 导入：`from torch.utils.data import DataLoader`

- **作用**：构建可迭代的数据装载器。在训练模型时使用此函数，用来把训练数据分成多个小组 ，每一次for循环（每一次enumerate）、每一次iteration都抛出一组batch_size大小的数据 

- **使用方法**：

  ```python
  torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, drop_last=False)
  ```

  - **dataset** (*Dataset*) – 加载数据的数据集，决定数据从哪读取以及如何读取
  - **batch_size** (*int*, optional) – 每个batch加载多少个样本，即一次摸多少张牌(默认: 1)。
  - **shuffle** (*bool*, optional) – 设置为`True`时会在每个epoch重新打乱数据，即下次摸牌时重新洗牌(默认: False).
  - **sampler** (*Sampler*, optional) – 定义从数据集中提取样本的策略，即生成index的方式，可以顺序也可以乱序
  - **num_workers** (*int*, optional) – 用多少个子进程加载数据：子进程越多加载越快，但有时会出错。0表示数据将在主进程中加载(默认: 0)
  - **drop_last** (*bool*, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

- **实例**：

  ```python
  train_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ])),
          batch_size=200, shuffle=True)
  # 或
  train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
  # 或
  
  ```

  - 

- 相关概念

  - **Epoch**： 所有训练样本都已输入到模型中，称为一个Epoch
  - **Iteration**： 一批样本输入到模型中，称为一个Iteration
  - **Batchsize**： 一批样本的大小， 决定一个Epoch有多少个Iteration

### 读取数据逻辑思路

data_dir（数据地址）-->Dataset-->DataLoader

![](E:\学习笔记\pic\pytorch基础1.png)

### 实例

实战

```python
from PIL import Image
img_path = "E:\\pytorch\\project\\pythonProject1\\hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
img.size
Out[5]: (768, 512)
img.show()
dir_path = "hymenoptera_data/train/ants"
import os
img_path_list = os.listdir(dir_path) # listdir：把所有文件dir变成列表list的形式
img_path_list[0]
Out[10]: '0013035.jpg'

```

- 注：一些关于os库的操作

  - `os.listdir(dir_path)`：把dir_path路径下所有文件dir变成列表list的形式

  - `os.path.join(root_dir,label_dir)`：把两个路径进行拼接

```python
from torch.utils.data import Dataset
from PIL import Image
import os # 一个库，可以获取所有图片的地址


class MyData(Dataset):
    def __init__(self, root_dir, label_dir): # 初始化函数，为class提供一个全局变量，为后面的函数提供它们所需要的量
        self.root_dir = root_dir # train文件夹地址
        self.label_dir = label_dir # ants
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path) # 获得所有ants图片名称列表 
        # self.：指定了类当中的全局变量，后面的函数也可以使用，相当于java里的this

    def __getitem__(self, idx):
        img_name = self.img_path[idx] # 获取具体某张图片的名称
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name) # 获取所有ants图片地址列表
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path) # 返回列表长度


# 创建类的实例
root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

train_dataset = ants_dataset + bees_dataset # 数据集的拼接
```

### 如何使用torchvision中的数据集



## 神经网络

> 参考教程：[torch.nn官方文档](https://pytorch.org/docs/stable/nn.html)、[60分钟快速入门PyTorch](https://zhuanlan.zhihu.com/p/66543791)、[最基础的pytorch神经网络代码](https://zhuanlan.zhihu.com/p/350512361)

### 0. 导入的包

```python
import torch.nn as nn
import torch.nn.functional as F
```

### 1. 容器

> 卷积层、池化层等各种层都是需要往里填充的东西

- `Module`：神经网络的类；它给所有神经网络提供了模板，我们使用时需要继承

  - init()：初始化函数，在其中需要完成继承、搭建网络的工作
  - forward(x)：前向传播函数，传入的x经过各个层的变换再输出

  ```python
  class Model(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(1, 20, 5)
          ...
  
      def forward(self, x):  # 是__call__方法的实现，调用Module实例直接会调用该函数
          x = F.relu(self.conv1(x))
          ...
          return x
      
  # 打印网络结构
  module = Module()
  print(module)  
  
  # 创建实例
  input = torch.tensor(1,2)
  output = module(input)
  ```

### 2. 层

> - 注意：torch.nn与torch.nn.functional中的层的用法不同，此处写的是torch.nn封装好的方法
>
> - 如果要求输入的数据必须是四维（N, C, H, W）时，进行torch.reshape变形

#### 常用的层

##### ① 卷积层

- 二维卷积层：`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)`
  
  - `in_channels`：输入四维张量[N,C,H,W]中的C，即输入张量的channel数；如RGB图像的输入层数为3
  
  - `out_channels`：期望的四维输出张量的channel数（即kernel_size的channel数）
  
  - `kernel_size`：卷积核的大小，即滑动窗口的大小；一般使用5x5或3x3大小的卷积核，即`kernel_size`=5或3
  
  - `stride`：步长
  
  - `padding`：`padding`即图像填充，后面的int型常数代表填充的多少（行数、列数），默认为0
  
  - **注**：输入的H和W取决于输入本身尺寸，而输出的H和W由计算得到
  
    ![](E:\学习笔记\pic\pytorch7.png)
  
    （上图中dilation默认为1）

##### ② 池化层（下采样）

- 二维最大池化层：`nn.MaxPool2d(kernel_size, stride, ceil_mode=False)`

  > 目的：保留数据特征的同时减小数据量

  - `kernel_size`：做最大池化的窗口大小
  - `stride`：步长，此处默认值不是1，而是kernel_size大小
  - `ceil_mode`：若为True，则保存不足kernel_size尺寸的值；否则不保存；默认为False

- **注**：池化层只改变H与W的值，不改变N与C

##### ③ 非线性激活层

> 目的：给神经网络引入非线性特征

- ReLU激活函数层：`nn.ReLU(inplace=False)`
  - `inplace`：若为True则输出直接替换输入，若为False则新建一个变量储存输出，默认为False
- Sigmoid激活函数层：`nn.Sigmoid(inplace=False)`

##### ④ 正则化层

- 正则化层可以提高神经网络的训练速度，不常使用

##### ⑤ 线性层

- 原理：g1=k11\*x1+b11，o1=k21\*g1+b21, ...

  下图中包含了两层线性层

  <img src="E:\学习笔记\pic\pytorch8.png" style="zoom:50%;" />

- 全连接层：`nn.Linear(in_features, out_features, bias=True)`

  - `in_features`：输入二维张量的大小
  - `out_features`：输出二维张量的大小，也即该全连接层的神经元个数

- 注：四维张量输入前需要先用`torch.reshape`或`torch.flatten`或`nn.Flatten`(类内)展平，确保输入为一维

  ```python
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)  # (3x32x32)->(32x32x32)
         	...
          self.flatten = nn.Flatten()  # 关键步骤
          self.linear1 = nn.Linear(64*4*4, 64)
          self.linear2 = nn.Linear(64, 10)
  ```

##### ⑥ softmax层

- 把上述层的结果输入，输出即可得到概率

##### ⑦ Dropout层

- 随机把一些元素按照p的概率变为0，以防止过拟合

- `nn.Dropout`

##### ⑧ 其他层

- ？？BN层？？

- `nn.Sequential`：可以把多个层封装到一起，定义自己的网络层

  ```python
  # 在__init__(self)中写：
  self.model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
          )
  
  # 在forward(self, x)中写：
  x = self.model(x)
  ```


- Recurrent层、Transformer层、Sparse层在具体的网络中才有应用

#### 使用torchvision中的现有模型

> 官网提供的模型：[torchvision.models](https://pytorch.org/vision/stable/models.html)

以常用网络vgg16为例

- 使用用法：

  ```
  vgg16 = torchvision.models.vgg16(pretrained=True)
  print(vgg16)
  ```

  - pretrain：True则下载参数（训练好的），False则只是加载网络模型，参数是默认的；即两者weight参数不同

  - 会发现最后一层输出有1000个分类，若用于CIFAR10的10分类需要作以调整：在vgg16后再加其他网络

- 在现有网络模型中添加层的方法：

  ```python
  vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
  vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
  ```

- 在现有网络模型中修改层的方法：

  ```python
  vgg16_true.classifier[6] = nn.Linear(4096, 10)
  ```

  

### 3. 损失函数

- 目的：用来衡量模型输出与实际输出的差距，其值越小越好；为反向传播提供目标函数

- 根据需求去使用，只需要注意输入怎样输出怎样

- 用法（以L1Loss为例）：`L1Loss(空)`

  ```python
  loss = L1Loss()  # 创建函数实例
  result = loss(inputs, targets)  # 计算损失函数
  ```

  - 输入的inputs、targets数据必须为四维tensor，若不是需要先torch.reshape

- 常用损失函数：

  - `MSELoss`

  - `CrossEntropyLoss`：交叉熵

    - 原理：

      ![](E:\学习笔记\pic\pytorch9.png)

    - 适用：分类问题

### 4. 反向传播

> [反向传播详细讲解](https://zhuanlan.zhihu.com/p/261710847)

- 目的：更新网络权重（即各个节点参数）



### 5. 优化器

#### 作用

实现参数自动优化

#### 分类

##### ① 最基础：随机梯度下降优化器SGD

##### ② 自适应梯度优化器AdaGrad

- 能够对每个不同的参数调整不同的学习率，对频繁变化的参数以更小的步长进行更新，而稀疏的参数以更大的步长进行更新
- 与SGD的核心区别：计算更新步长时，增加了分母：梯度平方累计和的平方根

##### ③ RMSProp

##### ④ 最优秀：Adam

- 结合AdaGrad和RMSProp两种优化算法的优点
- 对梯度的一阶矩估计（即梯度的均值）和二阶矩估计（即梯度的未中心化的方差）进行综合考虑，计算出更新步长

#### 优化器的创建（以SGD为例）

```python
optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=0.9)
```

- `net.parameters`：获取网络实例net中的参数
- `lr`：学习率（通常取0.01）
  - 学习率较小时，收敛到极值的速度较慢；
  - 学习率较大时，容易在搜索过程中发生震荡。
- `momentum`：动量

#### 优化器的使用

````python
optimizer.zero_grad()  # 清空梯度
output = net(input)    # 前向传播
loss = criterion(outputs, labels) #计算Loss
loss.backward()        # 反向传播
optimizer.step()       # 更新参数
````

- 清空梯度：所有梯度归零/重置，为下次反向传播做好准备（必须显式执行）
- 前向传播
- 计算Loss
- 反向传播：计算随机梯度进行反向传播
- 更新参数：执行一次梯度下降（以SGD为例）

### 6. 输出评价参数

> 不仅可以print，也可以在tensorboard中画图

- 每次训练的损失函数loss、每次epoch中的损失函数loss求和、loss曲线

- 训练集上的分类模型正确率

  ```python
  outputs = torch.tensor([0.1, 0.2],[0.05, 0.4])
  # argmax()分类函数的用法
  print(outputs.argmax(0))  # 0为纵向对比，1为横向对比
  >> tensor([0, 1])  # 0.1>0.05，分类为0；0.2<0.4，分类为1
  print(outputs.argmax(1))  # 0为纵向对比，1为横向对比
  >> tensor([1, 1])  # 0.1<0.2，分类为1；0.05<0.4，分类为1
  
  preds = outputs.argmax(1)  # 预测分类结果
  tragets = torch.tensor([0, 1])  # 实际分类结果
  print((preds == targets).sum())  # 计算出对应位置相等的个数
  ```

- 每N轮训练所用的训练时间

  ```python
  import time
  
  start = time.clock()
  end = time.clock()
  print("截至到目前的运行时间：{}s".format(end - start))
  ```

  

### 7. 神经网络的保存与读取

#### 两种保存读取方式

**方式1**：既能保存网络模型的结构，还能保存网络模型的参数

- 保存：

  ```python
  torch.save(vgg16, "vgg16_method1.pth")  # 网络实例名称，文件名

- 读取：

  ```python
  model = torch.load("vgg16_method1.pth")
  print(model)
  ```

**方式2**：（官方推荐）只能保存网络参数，但占用空间小

- 保存：

  ```python
  torch.save(vgg16.state_dict(), "vgg16_method2.pth")
  ```

  - 本质：把vgg的参数保存成一种字典格式

- 读取：

  ```python
  # 构建网络结构，然后直接引入参数
  vgg16 = torchvision.models.vgg16(pretrained=False)
  vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  
  
  # 或：只引入参数
  model = torch.load("vgg16_method2.pth")
  ```

**一些陷阱**：

- 保存：

  ```python
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
  
      def forward(self, x):
          x = self.conv1(x)
          return x
  
  net = Net()
  torch.save(net, "net_method1.pth")
  ```

- 错误读取：

  ```python
  model = torch.load('net_method1.pth')
  print(model)
  # 会报错：Can't get attribute 'Net'，说明还是需要复制网络结构，但是不需要创建实例
  ```

- 正确读取：

  ```python
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
  
      def forward(self, x):
          x = self.conv1(x)
          return x
  
  
  model = torch.load('net_method1.pth')
  print(model)
  ```


### 8. 使用训练好的神经网络进行预测

```python
# 读取预处理数据
image_path = "../..."
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchcision.transforms.ToTensor()])

image = transform(image)

# 加载模型
class Net(nn.Module)
	...
	
model = torch.load('net_n.pth') # 不要用net_1.pth，只训练了一轮，预测不准确
print(model)

# 把数据输入模型，得到输出
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
>>> tensor([[1.1429, -8.5345, 8.3514, 4.1395, 1.0852, ...]])  # 十个类别对应的概率
print(output.argmax(1))
>>> tensor([5])  # 预测出的类别

```



### 9. 模型逻辑思路

0. **创建主体函数，读取训练数据集与测试数据集**（数据来自MNIST）

   ```python
   def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
                 log_interval=10): # 以下所有部分均为该函数内容
   ```

   ```python
       train_loader = DataLoader(  # torchvision中datasets中所有封装的数据集都是torch.utils.data.Dataset的子类
           datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
           batch_size=batch_size, shuffle=True)
   
       test_loader = DataLoader(
           datasets.MNIST('../data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])),
           batch_size=batch_size, shuffle=True)
   ```

1. **创建自定义Net类**，继承nn.Module

   ```python
   class Net(nn.Module)
   ```

   1. `__init__()`：先super()，再搭建网络架构

      ```python
      import torch.nn as nn
      import torch.nn.functional as F
      
      class Net(nn.Module):
          def __init__(self):
              super(Net, self).__init__()
              self.fc1 = nn.Linear(28 * 28, 200)
              self.fc2 = nn.Linear(200, 200)
              self.fc3 = nn.Linear(200, 10)
      ```

      - 其中，`fc`为全连接层，若创建二维卷积层则使用`self.conv1 = nn.Conv2d(...)`

   2. `forward()`：定义数据如何在网络中流动；该方法覆盖基类中的一个伪方法

      ```python
      def forward(self, x):
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return F.log_softmax(x)
      ```

      - 其中，输入数据`x`为主要参数，`F.relu()`即对该层中的节点应用`ReLU`激活函数；最后一层不用`ReLU`激活函数，而是返回一个`log_softmax`函数进行激活。

2. **创建自定义Net类的实例net**，并打印net得到网络结构

   ```python
   net = Net()
   print(net)
   
   # 得到下列输出
   Net (
   (fc1): Linear (784 -> 200)
   (fc2): Linear (200 -> 200)
   (fc3): Linear (200 -> 10)
   )
   ```

3. **设置优化器与损失准则**，训练网络

   ```python
   # 创建一个随机梯度下降（stochastic gradient descent）优化器
   optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
   # 创建一个损失函数
   criterion = nn.NLLLoss()
   ```

   - 在**随机梯度下降优化器SGD**中指定学习率=0.01，momentum=0.9，并使用`.parameters()`方法将网络中的所有参数提供给优化器
   - 此处将损失准则设置为了**负对数似然损失**（与神经网络的log softmax输出相结合，为我们的10个分类类别提供了等价的交叉熵）

4. **训练网络**，运行主训练循环

   ```python
   # 运行主训练循环
   for epoch in range(epochs):
       for batch_idx, (data, target) in enumerate(train_loader):
           data, target = Variable(data), Variable(target)  # 将data与target转换为PyTorch变量
           data = data.view(-1, 28*28)  # 将数据大小从 (batch_size, 1, 28, 28) 变为 (batch_size, 28*28)
           optimizer.zero_grad()  # 清空梯度
           net_out = net(data)  # 前向传播
           loss = criterion(net_out, target)  # 计算Loss
           loss.backward()  # 反向传播
           optimizer.step()  # 更新参数
           if batch_idx % log_interval == 0:  # 迭代到一定次数时，打印一些结果
               print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                       epoch, batch_idx * len(data), len(train_loader.dataset),
                              100. * batch_idx / len(train_loader), loss.data[0]))
   ```

5. **测试网络**

   ```python
   # 运行测试循环
   test_loss = 0
   correct = 0
   for data, target in test_loader:
       data, target = Variable(data, volatile=True), Variable(target)  # 变量类型转换
       data = data.view(-1, 28 * 28)  # 数据尺寸转换
       net_out = net(data)  # 前向传播
       test_loss += criterion(net_out, target).data[0]  # 对批处理损失求和
       pred = net_out.data.max(1)[1]  # 得到有最大log概率判断结果的下标
       correct += pred.eq(target.data).sum()  # 将模型判断结果与实际值对比，一致则取1，并累加
   
   test_loss /= len(test_loader.dataset)
   print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
   ```

6. **main函数**

   ```python
   if __name__ == "__main__":
           create_nn()
   ```

### 10. 利用GPU训练

#### 方式1：用自己电脑上的GPU

**网络实例**、**损失函数**、**数据（imgs数据、labels数据）**

找到这三种数据，对其调用cuda再返回即可

```python
if torch.cuda.is_available():
    net = net.cuda()
    
if torch.cuda.is_available():
    loss_fun = loss_fun.cuda()
    
# 下面这个要写两次，在训练循环与测试循环中各写一次
if torch.cuda.is_available():
    imgs = imgs.cuda()
    labels = labels.cuda()
```

#### 方式2：还是用自己电脑上的GPU

```python
# 选择一种
device = torch.device("cpu")
device = torch.device("cuda")：用第一张显卡
device = torch.device("cuda:1")：用第二张显卡
# 或直接写成
device = torch.devide("cuda" if torch.cuda.is_available() else "cpu")

net = net.to(device)
# 或
net.to(device)

loss_fun = loss_fun.to(device)
# 或
loss_fun.to(device)

# 下面这个要写两次，在训练循环与测试循环中各写一次，且必须赋值
imgs = imgs.to(device)
labels = labels.to(device)
```



#### 方式3：没有GPU，用Google的GPU

Google Colab网站：https://colab.research.google.com/

新建笔记本，即可写代码运行，类似jupyter

修改-笔记本设置中，选择加速方式为GPU，每周可免费使用30h

Colab提供的显卡数据：

![](E:\学习笔记\pic\pytorch10.png)

注：感叹号+代码即表示在终端输入
