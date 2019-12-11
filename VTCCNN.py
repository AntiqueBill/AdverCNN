# 目标是在无干扰的情况下训练一个CNN模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from DataSet import load_train


# 加载数据
train_loader = load_train('./data/train/train0.mat', 128)

N = 62000  # 训练数据行数
C = 3000
classNum = 31
matfn = 'F:/train/train14.mat'
data = h5py.File(matfn, 'r')

train_x = np.transpose(data['train_x'])
train_y = np.transpose(data['train_y'])
print("train_x.shape:", train_x.shape)
print("train_y.shape:", train_y.shape)


batch_size = 128
nb_classes = 10
nb_epoch = 100
momentum = 0.3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(1, 30, 9),
            nn.ELU(),
            nn.Conv1d(30, 25, 9, 3),
            nn.ELU(),
            nn.MaxPool1d(3, 2),
            nn.Conv1d(24, 20, 9, 3),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(3260, 200),
            nn.Linear(200, 120),
            nn.Linear(120, 60),
            nn.Linear(60, 40),
            nn.Linear(40, 31)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer(x)
        x = self.softmax(x)
        return x


net = CNN()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=momentum)


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

PATH = './VTCCNN.pth'
torch.save(net.state_dict(), PATH)
