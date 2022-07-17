import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    def __init__(self, p):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 为保证网络深度, 在卷积层中加入padding
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.conv9 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        # # 此时大小为3x3x512, 直接进入3x3卷积核后转为1x1
        # self.pool4 = nn.MaxPool2d(kernel_size=(3, 3))
        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(3 * 3 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool1(F.relu(x))
        x = self.conv3(x)
        # x = F.relu(x)
        x = self.conv4(x)
        x = self.pool2(F.relu(x))
        x = self.conv5(x)
        # x = F.relu(x)
        x = self.conv6(x)
        # x = F.relu(x)
        x = self.conv7(x)
        # x = F.relu(x)
        x = self.conv8(x)
        x = self.pool3(F.relu(x))
        # x = self.conv9(x)
        # # x = F.relu(x)
        # x = self.conv10(x)
        # # x = F.relu(x)
        # x = self.conv11(x)
        # # x = F.relu(x)
        # x = self.conv12(x)
        # x = self.pool4(F.relu(x))
        x = torch.flatten(x, 1)
        x = self.fc1(F.relu(x))
        x = self.dropout(x)
        x = self.fc2(F.relu(x))
        x = self.dropout(x)
        x = self.fc3(F.relu(x))
        return x
