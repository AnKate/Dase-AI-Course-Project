# 代码注释生成
第四次实验：代码注释生成

使用预训练模型CodeBERT和CodeT5，在Ruby数据集上进行微调和测试；使用微调后的CodeT5模型在本次实验代码上生成注释。

最终在Ruby测试集上的BLEU得分为：**0.1324**（CodeBERT）、**0.2562**（CodeT5），**仅供参考**



## Setup

实验代码在ipynb中编写，若希望在本地运行，则需要安装以下依赖：

- transformers==4.20.1
- torch==1.12.0
- datasets==2.3.2

在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

 

## Repository Structure 

本仓库的文件结构如下：

```
|-- CodeBERT.ipynb	# CodeBERT预训练和测试代码
|-- CodeT5.ipynb	# CodeT5预训练和测试代码
|-- Summarization.ipynb	# 读取本次实验代码并生成注释
|-- README.md
|-- requirements.txt
```



## Usage

按照顺序运行对应的.ipynb文件即可。

对于Summarization.ipynb，需要首先将实验代码以**.py文件**的形式保存，若使用.ipynb生成的.py文件，则需要**注释掉所有非函数部分**，否则会在import时执行全部代码，花费较长时间。



## Reference

实验代码中部分参考自以下链接：

- https://github.com/salesforce/CodeT5
- https://github.com/microsoft/CodeBERT
- https://huggingface.co/course/chapter2/1?fw=pt

