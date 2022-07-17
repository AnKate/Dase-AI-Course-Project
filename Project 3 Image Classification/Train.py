import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
from LeNet import *
from AlexNet import *
from ResNet import *
from VGGNet import *


# 训练用的函数
def train(model, epoch_num, learning_rate, dropout):

    if model == "LeNet":
        net = LeNet()
    elif model == "AlexNet":
        net = AlexNet(dropout)
    elif model == "ResNet":
        net = ResNet18()
    elif model == "VGGNet":
        net = VGGNet(dropout)

    train_data = torchvision.datasets.MNIST(
        root='./MNIST',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    test_data = torchvision.datasets.MNIST(
        root='/MNIST',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    use_gpu = torch.cuda.is_available()
    # print(use_gpu)
    if use_gpu:
        device = torch.device("cuda:0")
        net = net.to(device)

    for epoch in range(epoch_num):
        running_loss = 0
        acc = 0
        cnt = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc += (outputs.argmax(dim=1) == labels).sum().item()
            cnt += labels.shape[0]
            running_loss += loss.item()
        print('[%d] loss: %.4f acc: %.3f' % (epoch + 1, running_loss / cnt, acc / cnt))

    print('Training Finish.')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, ans = data
            inputs = inputs.to(device)
            ans = ans.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted.tolist())):
                total += ans.size(0)
                correct += (predicted == ans).sum().item()

    # print(total)
    # print(correct)
    print('Accuracy: %.3f%%' % (100 * correct / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="选择使用的模型")
    parser.add_argument('--epoch', type=int, help="训练的epoch个数")
    parser.add_argument('--lr', type=float, help="选择学习率")
    parser.add_argument('--dropout', type=float, help="dropout的概率")
    args = parser.parse_args()

    train(args.model, args.epoch, args.lr, args.dropout)
