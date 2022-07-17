import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层C1: 输入通道1, 输出通道6, 卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        # 池化层S2: 窗口大小2x2, 输入与输出通道均为6
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 卷积层C3: 输入通道6, 输出通道16, 卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        # 池化层S4: 窗口大小2x2, 输入与输出通道均为16
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 卷积层C5: 输入通道16, 输出通道120, 卷积核大小5x5
        # self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.fc0 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层F6: 神经元个数设置为84
        self.fc1 = nn.Linear(120, 84)
        # 全连接层(输出层): 最后分为10类
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        # x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc0(F.relu(x))
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        return x
