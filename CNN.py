import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./mnist_dataset', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./mnist_dataset', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=(2, 2), stride=(2, 2))

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, kernel_size=(2, 2), stride=(2, 2))

        X = X.view(-1, 16 * 5 * 5)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time

start_time = time.time()

total_epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(total_epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        predicated = torch.max(y_pred, 1)[1]
        batch_corr = (predicated == y_pred).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 600 == 0:
            print(f'Epoch: {i} Batch: {b} loss: {loss.item()}')

    train_correct.append(loss)
    train_correct.append(trn_corr)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicated = torch.max(y_val.data, 1)[1]
            tst_corr += (predicated == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

current_time = time.time()
total_time = current_time - start_time
print(f'Training Time: {total_time / 60} minutes')

# Save our model
torch.save(model.state_dict(), 'CNN_Model_Pytorch.pt')
