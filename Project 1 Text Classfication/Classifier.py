import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import doc2vec

feature_num = 12000


# MLP部分, 使用torch实现
# 增添线性层的层数对分类结果的影响不大, 3~4层即可
# dropout会使得在训练集上的表现下降, 故不使用
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_num, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 10)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


# 自定义测试集的Dataset类, 继承自torch的Dataset类
class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":

    # 预处理部分
    # 读取数据, 存储label和文本
    f = open('./train_data.txt', 'r')
    cnt = 0
    labels = []
    sentences = []
    lines = f.readlines()
    # 对文本进行分词
    for line in lines:
        temp = json.loads(line)
        labels.append(temp['label'])
        sentences.append(temp['raw'])
        # temp_list = temp['raw'].split(' ')
        # words = []
        # for word in temp_list:
        #     if word != '':
        #         words.append(word)
        # cnt += len(words)
        # sentences.append(words)

    # # Doc2Vec的向量化方法
    # # Doc2Vec转化为文档向量, 需要进行tag标记
    # train_X = []
    # tag = 0
    # for sentence in sentences:
    #     train_X.append(doc2vec.TaggedDocument(sentence, tags=[tag]))
    #     tag += 1
    #
    # # 使用Doc2Vec转换为向量, vector_size是向量化后的维度
    # # 经测试, 维度对正确率提升的帮助不大, 但会明显地降低运行效率
    # model = doc2vec.Doc2Vec(train_X, vector_size=2000)
    # # 进行训练, epoch越大正确率越高, 最后稳定在80%左右
    # # 不对Doc2Vec模型本身进行训练, 则会导致分类结果正确率低, 训练的目的是使得文档向量趋于稳定
    # model.train(train_X, total_examples=model.corpus_count, epochs=60)

    # tf-idf的向量化方法
    # TfidVectorizer能够在训练集上生成维度为max_features的文档向量
    model = TfidfVectorizer(max_features=feature_num)
    model_fit = model.fit_transform(sentences)
    matrix = model_fit.toarray()
    # 由于tf-idf的向量化方法是基于在语料库中较为重要的词语(feature)计算得出的
    # 而测试集与训练集的feature有所出入, 经测试后发现两者交集仅有7000左右的单词
    # MLP是基于训练集调整的参数, 为保证参数依然适用, 需要将训练集中的feature导出, 作为新模型的训练词典
    voc1 = model.get_feature_names_out()
    vocab_dict = {}
    idx = 0
    # 必须采用dict类型
    for i in voc1:
        vocab_dict[i] = idx
        idx += 1

    # print(vocab_dict)

    dataset = []
    for i in range(len(matrix)):
        dataset.append((matrix[i], labels[i]))

    # print(dataset)

    # 划分数据集
    test_num = 2000    # 使用与测试集同样的验证集大小
    train_num = 6000    # 训练集大小
    train_set, test_set = random_split(dataset, [train_num, test_num])

    train_list = list(train_set)
    test_list = list(test_set)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(train_list)):
        X_train.append(train_list[i][0])
        y_train.append(train_list[i][1])
    for i in range(len(test_list)):
        X_test.append(test_list[i][0])
        y_test.append(test_list[i][1])

    X_train = torch.from_numpy(np.array(X_train).astype(np.float32))
    y_train = torch.from_numpy(np.array(y_train))
    X_test = torch.from_numpy(np.array(X_test).astype(np.float32))
    y_test = torch.from_numpy(np.array(y_test))

    loader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)
    loader_test = DataLoader(TensorDataset(X_test, y_test), batch_size=100)

    # MLP部分
    net = Net()
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 学习率尝试过1e-2以及更低, 但训练的loss降低速度缓慢, 需要增大epoch的数量
    # 采用带动量的SGD作为优化器
    optimizer = optim.SGD(net.parameters(), lr=8e-2, momentum=0.9)

    # 训练
    # 进行15个epoch的训练
    print("————————————————训练————————————————\n")
    for epoch in range(15):
        running_loss = 0.0
        for i, data in enumerate(loader_train, 0):
            inputs, labels = data
            labels = labels.long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(i)
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    # 验证
    correct = 0
    total = 0
    print("————————————————验证————————————————\n")
    with torch.no_grad():
        for data in loader_test:
            inputs, labels = data
            labels = labels.long()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 2000 test texts: %d %%' % (
            100 * correct / total))
    # print(correct)

    # 基于全部数据集再次进行训练
    total_X = []
    total_y = []
    for i in range(len(dataset)):
        total_X.append(dataset[i][0])
        total_y.append(dataset[i][1])
    total_X = torch.from_numpy(np.array(total_X).astype(np.float32))
    total_y = torch.from_numpy(np.array(total_y))

    loader_total = DataLoader(TensorDataset(total_X, total_y), batch_size=100, shuffle=True)
    print("———————————————完整训练———————————————\n")
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(loader_total, 0):
            inputs, labels = data
            labels = labels.long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    # 对测试集进行分类, 并写入文件
    f = open('./test.txt', 'r')
    lines = f.readlines()
    test = []
    for line in lines:
        line = line.replace('\n', '')
        idx = line.index(',')
        test.append(line[idx + 2:])

    test = test[1:]

    new_model = TfidfVectorizer(max_features=feature_num, vocabulary=vocab_dict)
    new_model_fit = new_model.fit_transform(test)
    new_matrix = new_model_fit.toarray()
    # voc2 = set(new_model.get_feature_names_out())
    # print(len(voc2 & voc1))
    test_X = []
    for i in range(len(test)):
        test_X.append(new_matrix[i])

    X = torch.from_numpy(np.array(test_X).astype(np.float32))
    # print(X)
    Loader = DataLoader(TestDataset(np.array(test_X).astype(np.float32)))

    out = open('submit_sample.txt', 'w', encoding='utf-8')
    out.write('id, pred\n')
    i = 0
    with torch.no_grad():
        for data in Loader:
            # print(data)
            inputs = data
            # labels = labels.long()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            out.write(str(i) + ', ' + str(predicted.item()) + '\n')
            i += 1
