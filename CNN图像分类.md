# CNN图像分类

## 1. 项目介绍

基于pytorch实现一个卷积神经网络，

完成CIFAR-10图像分类任务。

## 2. 数据集

CIFAR-10

10个类别：

```python
"飞机",
"汽车",
"鸟",
"猫",
"鹿",
"狗",
"青蛙",
"马",
"船",
"卡车"
```

## 3. 模型结构

卷积层（Conv2d）：提取图像特征

ReLU激活函数：引入非线性

最大池化层（MaxPooling）：降低特征图尺寸

卷积层（Conv2d）：进一步提取高级特征

最大池化层（MaxPooling）：压缩特征信息

全连接层（Fully Connected）：完成分类预测

## 4. 实验环境

操作系统：Windows 11

编程语言：Python 3.13

深度学习框架：PyTorch 2.13

计算设备：CPU（Intel Core i5-13500H）

内存：16GB

## 5. 训练结果

测试正确率：68.39%

损失曲线：

![Figure_1](C:\Users\尹宇曼mm\Desktop\images\Figure_1.png)

## 6. 预测展示

![Figure_cat](C:\Users\尹宇曼mm\Desktop\images\Figure_cat.png)



