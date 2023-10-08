##%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self, in_features=4, out_features=3, h1_units=10, h2_units=20):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1_units)
        self.fc2 = nn.Linear(h1_units, h2_units)
        self.out = nn.Linear(h2_units, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_data = pd.read_csv(url)

print(my_data.head())
print(my_data.describe())

my_data['variety'] = my_data['variety'].replace('Setosa', 0.0)
my_data['variety'] = my_data['variety'].replace('Versicolor', 1.0)
my_data['variety'] = my_data['variety'].replace('Virginica', 2.0)
print(my_data)

X = my_data.drop('variety', axis=1)
y = my_data['variety']

# convert X and y to numpy array
X = X.values
y = y.values

# Train and Test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert X_train, X_test to Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# convert y_train, y_test = Tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = Model(X_train, y_train, 20, 10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
losses = []



for epoch in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.detacj().numpy())

    if epoch % 10 ==0:
        print(f'Epoch: {epoch} amd loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





##%

