# 文本分类
第一次实验：文本分类

使用多层感知机对向量化后的文档进行分类。

最终在验证集上的正确率为**90%左右**，**仅供参考**



## Setup

运行实验代码需要安装以下依赖：

- scikit-learn==1.0.2
- torch==1.12.0

在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

 

## Repository Structure 

本仓库的文件结构如下：

```
|-- Classifier.py	# 实验代码
|-- train_data.txt	# 训练集
|-- test.txt	# 测试集
|-- submit_sample.txt	# 分类结果
|-- README.md
|-- requirements.txt
```



## Usage

在终端中输入

```shell
python Classifier.py
```

运行.py文件即可。

实验的训练集与测试集需要与实验代码放于同一目录下。



## Reference

实验代码中部分参考自以下链接：

- https://pytorch.apachecn.org/
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

