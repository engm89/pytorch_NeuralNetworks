import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import Linear, ReLU, Sigmoid, Tanh, Module, BCELoss
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from torch.optim import SGD
from torch.utils.data import Dataset, random_split, DataLoader

path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'

class fetchData(Dataset):
    def __init__(self, path):
        data = pd.read_csv(path, header=None)
        self.X = data.values[:, :-1]
        self.y = data.values[:, -1]

        self.X = self.X.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape(len(self.y), 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def split(self, ratio=0.5):
        size1 = round(ratio * len(self.X))
        size2 = len(self.X) - size1
        return random_split(self, [size1, size2])


def getData(path, ratio=0.5):
    obj = fetchData(path)
    train, test = obj.split(ratio)
    train = DataLoader(train, shuffle=True, batch_size=30)
    test = DataLoader(test, shuffle=True, batch_size=1000)
    return train, test


train, test = getData(path, 0.2)
print(train)


class NeuralNetwork(Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = Linear(input_size, 1, bias=False)
        self.activation1 = Tanh()


class MultiLayerPerceptron(Module):
    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = Linear(input_size, 10)
        kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        self.activation1 = ReLU()

        self.layer2 = Linear(10, 6)
        kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        self.activation2 = ReLU()

        self.layer3 = Linear(6, 1)
        xavier_uniform_(self.layer3.weight)
        self.activation3 = Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        x = self.layer3(x)
        x = self.activation3(x)

        return x

num_epochs= 100
def training(train_data, model):

    loss_func = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.85)
    for epoch in range(num_epochs):
        for (X,y) in train_data:
            optimizer.zero_grad()
            results = model(X)
            loss = loss_func(results, y)
            loss.backward()
            optimizer.step()
        print("At Epoch: ",epoch," The loss: ", loss.detach().numpy())

train, test = getData(path)
input_size = len(train.dataset[0][0])
model = MultiLayerPerceptron(input_size)
training(train, model)

def evaluation(test, model):
    for X,y in test:
        result = model(X)
        result = result.detach().numpy()
        y = y.detach().numpy()
        result = result.round()
        print("The Accuracy(Training) : ", round(accuracy_score(result, y), 2)*100, '%')
evaluation(test, model)

