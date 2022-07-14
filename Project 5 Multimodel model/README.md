# 多模态情感分析
第五次实验：多模态情感分析

构建一个多模态模型，对给定的图像+文本，实现情感三分类（positive、negative、neutral）

模型在验证集上的最终正确率：**62.6%**，**仅供参考**



## Setup

实验代码在ipynb中编写，若希望在本地运行，则需要安装以下依赖：

- transformers==4.20.1
- torch==1.12.0
- numpy==1.21.6
- pillow==7.1.2

在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

 

## Repository Structure 

本仓库的文件结构如下：

```
|-- data	# 文本+图像的数据集
|-- Multimodal.ipynb	# 完整的项目代码
|-- train.txt	# 训练集: 数据的文件名及其对应标签
|-- test_without_label.txt	# 测试集: 待分类的文件名
|-- test.txt	# 预测结果
|-- README.md
|-- requirements.txt
```



## Usage

打开Multimodal.ipynb，按照顺序执行即可。

实验数据集请置于/data/目录下，训练集与测试集请严格遵守实验指定的**guid,label**的格式。



## Reference

实验代码中部分参考自以下链接：

- https://zhuanlan.zhihu.com/p/402997033
- https://blog.csdn.net/weixin_36979214/article/details/108879684

