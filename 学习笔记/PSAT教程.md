# PSAT教程

## 〇、安装

- `PSAT`文件夹添加到MATLAB路径

- `matpower`文件夹放到MATLAB文件夹下的toolbox文件夹下
- 如果在命令行输入`psat`可以启动，则说明安装成功；否则可能是PSAT与MATLAB版本不匹配的问题

## 一、模块介绍

> [simulink模块介绍](https://www.bilibili.com/video/BV1Lg4y1z7g8/?vd_source=28f4ef64be559a1c9c4ab1cf2b8ee826)

- simulink画图模块

  ![](E:\学习笔记\pic\PSAT1.png)

- 从左到右、从上到下依次为：

  ![](E:\学习笔记\pic\PSAT2.png)

  - 连接模块
  - 潮流模块
  - OPF（最优潮流）&CPF（连续潮流）模块
  - 模拟故障元件
  - 负荷模块
  - 动力模块（同步发电机与异步发电机）
  - 控制模块（各种调速器、调节器等）
  - 调压变压器模块（有载调压等）
  - 柔性输电模块（静止无功补偿器等）
  - 风力发电模块
  - 其他模型（包括燃料电池模型、次同步振荡模型、光伏发电模型）
  - 测量单元（频率测量、相量测量单元PMU）