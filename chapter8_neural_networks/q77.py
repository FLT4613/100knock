# Ref:
# https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-dataset
# https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-dataloader

from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

train_x = torch.tensor(np.loadtxt('files/q7_train_x.txt')).float()
train_y = torch.tensor(np.loadtxt('files/q7_train_y.txt')).long()
valid_x = torch.tensor(np.loadtxt('files/q7_valid_x.txt')).float()
valid_y = torch.tensor(np.loadtxt('files/q7_valid_y.txt')).long()


def get_accuracy_score(pred, y):
    return torch.eq(torch.max(pred, 1).indices, y).sum().item() / len(y)


elapsed = []

for bs in tqdm([2 ** x for x in range(0, 7)]):
    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    model = nn.Sequential(*[torch.nn.Linear(300, 4)])
    op = optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    epoch = 10
    model.train()

    start = time()
    for i in tqdm(range(epoch)):
        for x, y in train_dl:
            train_pred = model(x)
            train_loss = loss(train_pred, y)
            op.zero_grad()
            train_loss.backward()
            op.step()

        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_x)
            valid_loss = loss(valid_pred, valid_y)

    elapsed.append((str(bs), (time() - start) / epoch))

plt.bar(*zip(*elapsed))
plt.xlabel('Batch Size')
plt.ylabel('Training time per epoch')
plt.show()
