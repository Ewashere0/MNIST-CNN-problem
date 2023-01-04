import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

# Prepare dataset
def prepareData(X=None, y=None, file_name=None):
    assert file_name is not None or (X is not None and y is not None)
    if file_name:
      train_data = np.loadtxt(file_name, delimiter=',')
      y = train_data[:, 0]
      X = train_data[:, 1:] / 255.

    #reshape X as [[[top_img_0], [middle_img_0], [bottom_img_0]], ..., [[top_img_n], [middle_img_n], [bottom_img_n]]]
    #(10000, 3, 28, 28)
    n, d = X.shape
    X = np.array([x.reshape((3, int((d/3)**(1/2)), int((d/3)**(1/2)))) for x in X])

    # Convert to PyTorch Tensors
    tensor_X = torch.Tensor(X)
    tensor_y = torch.LongTensor(y)

    dataset = TensorDataset(tensor_X, tensor_y)

    return dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.conv1b = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.conv1c = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv3a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.conv3b = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.conv3c = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        
        self.fc1 = nn.Linear(3*32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        a, b, c = torch.split(x, 1, dim=1) 

        a = self.pool(torch.relu(self.conv1a(a)))
        a = self.pool(torch.relu(self.conv3a(a)))

        b = self.pool(torch.relu(self.conv1b(b)))
        b = self.pool(torch.relu(self.conv3b(b)))

        c = self.pool(torch.relu(self.conv1c(c)))
        c = self.pool(torch.relu(self.conv3c(c)))

        x = torch.cat((a, b, c), dim=1)
        x = x.view(-1, 3*32 * 7 * 7)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        x = F.log_softmax(x, dim=1)

        return x

def train(model, train_data, lr, batch_size=16, momentum=0.95, decay=5e-4):
    device = "cuda"
    # training data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # training
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        model.train() # annotate model for training
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def learn(X, y):
  device = "cuda"
  model = CNN().to(device)
  train_dataset = prepareData(X, y)

  for epoch in range(25):
    lr=1e-2 if epoch < 20 else 1e-3
    train(model, train_dataset, lr=lr, batch_size=16, momentum=0.95, decay=5e-4)

  return model

def classify(Xtest, model):
  device = "cuda"
  n, d = Xtest.shape
  Xtest = np.array([x.reshape((3, int((d/3)**(1/2)), int((d/3)**(1/2)))) for x in Xtest])
  tensor_X = torch.Tensor(Xtest).to(device)

  out = model(tensor_X)
  pred = out.argmax(dim=1, keepdim=True)
  yhat = np.array(pred.cpu().view(-1))

  return yhat

#sample driver code
train_data = np.loadtxt('train.csv', delimiter=',')
y = train_data[:, 0]
X = train_data[:, 1:] / 255.

model = learn(X, y)

val_data = np.loadtxt('val.csv', delimiter=',')
yval = val_data[:, 0]
Xval = val_data[:, 1:] / 255.

x=10
yhat = classify(Xval, model)
print(np.sum(yval == yhat) / len(yval))
#0.7646