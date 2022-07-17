import torch
import torch.nn as nn
import torch.nn.functional as F


# 不含shortcut的残差单元, 包含两层卷积层
class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(output_channel)
        # 卷积层2不改变通道数
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = F.relu(output + x)
        return output


# 含有shortcut的残差单元, 相较于一般的残差单元多了一层1x1的卷积层以改变通道数
class ShortcutResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ShortcutResBlock, self).__init__()
        # 改变通道数用的卷积层
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(output_channel)
        # 常规的残差卷积层
        self.conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        output1 = self.conv1(x)
        output1 = self.bn1(output1)
        output2 = self.conv2(x)
        output2 = self.bn2(output2)
        output2 = F.relu(output2)
        output2 = self.conv3(output2)
        output2 = self.bn3(output2)
        output = F.relu(output1 + output2)
        return output


# 参考自ResNet18的网络, 但由于MNIST的图像过小, 减少了残差单元的层数
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # 卷积层: 输入通道1, 输出通道64, 卷积核大小5x5
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(64)
        # 池化层:窗口大小2x2, 步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 前两个残差单元的通道数均为64
        self.res1 = ResBlock(64, 64)
        self.res2 = ResBlock(64, 64)
        # shortcut的残差单元, 通道数改为128
        self.shortcut1 = ShortcutResBlock(64, 128)
        self.res3 = ResBlock(128, 128)
        self.shortcut2 = ShortcutResBlock(128, 256)
        self.res4 = ResBlock(256, 256)
        # 此时的数据维度为3x3x256, 故不再进入残差层
        # 进入线性层之前的池化层为平均池化层, 输出结果为1x1x256
        self.pool2 = nn.AvgPool2d((3, 3))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(F.relu(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.shortcut1(x)
        x = self.res3(x)
        x = self.shortcut2(x)
        x = self.res4(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
