import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, p):
        super(AlexNet, self).__init__()
        # 卷积层: 输入大小调整为28x28, 输入通道1, 输出通道96, 卷积核大小为5x5, 步长为1
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(5, 5), stride=1)
        # 池化层: 窗口大小2x2, 步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 卷积层: 输入大小为12x12, 输入通道96, 输出通道256, 卷积核大小为5x5, padding=2, 步长为1
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2)
        # 池化层: 窗口大小2x2, 步长为2
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 卷积层: 输入大小为6x6, 输入通道256, 输出通道384, 卷积核大小为3x3, padding=1, 步长为1
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1)
        # 卷积层: 输入大小为6x6, 输入通道256, 输出通道384, 卷积核大小为3x3, padding=1, 步长为1
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1)
        # 卷积层: 输入大小为6x6, 输入通道384, 输出通道256, 卷积核大小为3x3, padding=1, 步长为1
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1)
        # 池化层: 窗口大小2x2, 步长为2
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 全连接层: 输入为3x3x256, 神经元个数为1024
        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(3 * 3 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.pool3(F.relu(x))
        x = torch.flatten(x, 1)
        x = self.fc1(F.relu(x))
        x = self.dropout(x)
        x = self.fc2(F.relu(x))
        x = self.dropout(x)
        x = self.fc3(F.relu(x))
        return x
