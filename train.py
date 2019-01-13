import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import time
import os
from dataloader import dataset
from Net import myNet
import torch.functional as F

batchSize = 128


# 加载数据
def loadData():
    trainingData = dataset(root='trainning.csv',
                           transform=transforms.Compose([transforms.ToTensor()]))
    training = DataLoader(
        trainingData, batch_size=batchSize, shuffle=True, num_workers=0)

    testData = dataset(root='test.csv',
                       transform=transforms.Compose([transforms.ToTensor()]))
    test = DataLoader(
        testData, batch_size=batchSize, shuffle=True, num_workers=0)
    return training, test


# 训练模型
def training(model, epoch, trainloader, optimizer):
    model.train()
    loss = nn.NLLLoss()

    for n, (data, target) in enumerate(trainloader):
        data = data.type(torch.float)
        if torch.cuda.is_available():
            data = (data).to('cuda')
            target = (target).to('cuda')
            model.to('cuda')
        output = model(data)
        optimizer.zero_grad()
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        if n % 10 == 0:
            print('training epoch: %d loss:%.3f ' % (epoch + 1, loss_value.item()))
            predict = output.data.max(1)[1]
            number = predict.eq(target.data).sum()
            correct = 100 * number / batchSize
            print("\t", predict[0:20])
            print("\t", target[0:20])
            print('Accuracy:%0.2f' % correct, '%')


def train():
    model = myNet()
    outset = time.time()
    trainloader, testloader = loadData()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    for epoch in range(epochs):
        training(model, epoch, trainloader, optimizer)
    torch.save(model, 'mymodel2.pth')
    model = torch.load('mymodel2.pth')


#	print(time.time()-outset "has passed.")

if __name__ == '__main__':
    train()
