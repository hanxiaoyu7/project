教程：[pytorch深度学习快速入门](bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=28f4ef64be559a1c9c4ab1cf2b8ee826)（但是这个教程里的版本都太低了，注意不要跟着选择）

## 〇、安装

> - **安装重点**：所装的anacoda、python、cuda、pytorch以及自己的显卡GPU、驱动**型号一定要匹配**！不要盲目安装最新版本！90%的麻烦问题都是版本不匹配的锅！
> - 示例版本1：anaconda3 + python3.9 + cuda11.1 + pytorch1.9.0（可惜我的垃圾显卡配不上这个版本，无奈只好卸载重装版本2）
> - 示例版本2：anaconda3 + python3.6 + cuda9.2 + pytorch1.2.0
> - 具体怎么匹配会在教程里讲

### 1. 安装anaconda

- anaconda中一个package相当于一个工具包
- anaconda安装网址：[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)（官网收费）
- 我安装的是anaconda3-2022.10-windows版本，即安装时刻的最新版本

### 2. 切换环境

- 切换环境，即不同项目中不同版本pytorch的切换
- 在anacoda命令行依次输入下列命令：

 ```
conda create -n pytorch python=3.6 % 安装python3.6的环境，名为pytorch
conda activate pytorch % 切换到pytorch环境
pip list % 查看当前环境下有哪些包，会发现没有pytorch
 ```

- 其他一些有用的命令：

```
conda activate base % 切换到base环境
conda info --envs % 查看所有已安装的环境
conda remove -n pytorch --all % 删除该pytorch环境
conda clean --packages % 清除不用的包

conda uninstall pytorch 
conda uninstall libtorch % 这两条卸载pytorch（若用pip安装的pytorch则把conda改成pip）
conda uninstall -n base --all % 卸载base下的所有包
```

### 3. 安装pytorch

> 不用额外安装CUDA，就算你电脑不装cuda，只要装了cuda版本的pytorch就可以跑GPU了。啥意思呢，就是说pytorch那个cuda版本已经包含了跑GPU所需要的cuda核心模块，而电脑里单独装的cuda则更加全面，只装前者也可以，两者都有更好。当然前提是你的显卡本身要支持你装的pytorch-cuda版本。

- pytorch安装网址：[pytorch官网](https://pytorch.org/get-started/locally/)

- 型号选择：

  - 首先，查看自己的电脑的驱动版本以及支持的CUDA版本
  
    - 方法1：在命令行输入`nvidia-smi`，可以看到第一行的Driver Version与CUDA Version
  
      ![](E:\学习笔记\pic\pytorch安装2.png)
  
    - 方法2：在NVIDIA控制面板左下角的系统信息可以查看
  
    - 可以看到我的驱动最高只支持CUDA11.2
  
  - 其次，还要查看自己电脑的GPU型号的算力，由此选择合适的pytorch版本
  
    - 在[这里](https://developer.nvidia.com/cuda-gpus)可以查自己的GPU的算力
    - GPU算力与对应的pytorch版本可以在网上查到
  
  - 综合上述两者，选择安装的CUDA-pytorch版本：官网给的CUDA11.6和CUDA11.7对我的驱动来说版本太太太高了，于是我选择历史版本型号（见红色箭头），选择安装pytorch1.2.0对应的CUDA9.2
  
  ![](E:\学习笔记\pic\pytorch1.png)
  
  - 【重要】本来选择的用conda安装，但是持续出现安装某package时失败的报错，搜了一下好多人说用pip安装就好了，于是选择了**pip**（wheel下面的），竟然成功了！不懂为什么但是很开心（但是anaconda还有必要吗..?

![](E:\学习笔记\pic\pytorch2.png)

- 将对应的安装代码复制粘贴到命令行

```
# CUDA 9.2
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

- 检验pytorch是否安装成功：

  ```
  pip list % 有torch即为安装好了pytorch
  python % 进入python环境
  import torch % 若直接显示下一行则安装成功
  ```
  
- 检验CUDA是否安装正确并能被pytorch检测到：

  ```
  （接上一块代码）
  torch.cuda.is_available() % 若为True则成功
  ```

- 检验GPU能否正常使用：

  ```
  （接上一块代码）
  a=torch.Tensor([1,2])
  a=a.cuda()
  a
  ```

  - 这一步是关键！很容易出现报错：`RuntimeError: CUDA error: no kernel image is available for execution on the device`

  - 出错原因：显卡计算能力太低或pytorch版本太高；从pytorch1.3开始，即不再支持GPU算力在3.5及以下的GPU

    ![](E:\学习笔记\pic\pytorch安装1.png)

  - 解决方案：参考[这里](https://blog.csdn.net/qq_44159782/article/details/121951993)，我第一次也遇到这样的问题，于是卸载重装了低版本pytorch


### 4. 安装pycharm

- pycharm安装网址：[pycharm官网](https://www.jetbrains.com/pycharm/)

- 记住同样不要盲目下载最新版本，否则会出现各种问题。我下载的是2021.1.1社区版

  ![](E:\学习笔记\pic\pytorch4.png)

- 环境配置具体步骤见视频教程。注：若显示conda可执行路径为空，卸载当前版本pycharm去下载历史旧版本的pycharm。

  - 配置成功：

    ![](E:\学习笔记\pic\pytorch5.png)

- 在pytorch环境下安装jupyter的具体步骤见视频教程。(又失败了，搞不定，pip和conda安装都报错，放弃了)
