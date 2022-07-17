# 图像分类
第三次实验：图像分类

使用LeNet、AlexNet、ResNet、VGGNet对MNIST数据集进行分类。

最终正确率：**98.74%**（LeNet）、**99.11%**（AlexNet）、**99.34%**（ResNet）、**99.12%**（VGGNet），**仅供参考**



## Setup

运行实验代码需要安装以下依赖：

- torch==1.12.0

在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

 

## Repository Structure 

本仓库的文件结构如下：

```
|-- AlexNet.py	# Alexnet
|-- LeNet.py	# LeNet
|-- ResNet.py	# ResNet
|-- VGGNet.py	# VGGNet
|-- Train.py	# 训练函数
|-- README.md
|-- requirements.txt
```



## Usage

在终端中输入

```shell
python Train.py --model $Model --epoch $epoch_num --lr $learning_rate --dropout $dropout_p
```

运行.py文件。

其中：

- model参数指定使用的模型，需要在LeNet、AlexNet、ResNet、VGGNet中选取
- epoch参数指定训练的轮数
- lr参数指定训练学习率
- dropout参数指定dropout层神经元被舍弃的概率p（部分模型不存在dropout层）



## Reference

实验代码中部分参考自以下链接：

- https://zhuanlan.zhihu.com/p/116181964
- https://zhuanlan.zhihu.com/p/116197079
- https://zhuanlan.zhihu.com/p/116900199
- https://blog.csdn.net/weixin_43999691/article/details/117928537

