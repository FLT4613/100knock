from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

train_x = torch.tensor(np.loadtxt('files/q7_train_x.txt')).float()
train_y = torch.tensor(np.loadtxt('files/q7_train_y.txt')).long()
valid_x = torch.tensor(np.loadtxt('files/q7_valid_x.txt')).float()
valid_y = torch.tensor(np.loadtxt('files/q7_valid_y.txt')).long()

model = nn.Sequential(*[
    torch.nn.Linear(300, 4)
])


def get_accuracy_score(pred, y):
    return torch.eq(torch.max(pred, 1).indices, y).sum().item() / len(y)


op = optim.SGD(model.parameters(), lr=0.1)
celoss = nn.CrossEntropyLoss()

epoch = 3000
datapoints = []
for i in tqdm(range(epoch)):
    model.train()
    train_pred = model(train_x)
    train_loss = celoss(train_pred, train_y)
    op.zero_grad()
    train_loss.backward()
    op.step()

    model.eval()
    with torch.no_grad():
        valid_pred = model(valid_x)
        valid_loss = celoss(valid_pred, valid_y)

        datapoints.append([
            train_loss.item(),
            valid_loss.item(),
            get_accuracy_score(train_pred, train_y),
            get_accuracy_score(valid_pred, valid_y)
        ])


df = pd.DataFrame(datapoints)

ax_loss = plt.subplot(2, 1, 2)
ax_loss.set_title("Loss")
ax_loss.plot(range(epoch), df.iloc[:, 0], label='train',)
ax_loss.plot(range(epoch), df.iloc[:, 1], label='valid',)
ax_loss.legend()

ax_acc = plt.subplot(2, 1, 1)
ax_acc.set_title("Accuracy")
ax_acc.plot(range(epoch), df.iloc[:, 2], label='train',)
ax_acc.plot(range(epoch), df.iloc[:, 3], label='valid',)
ax_acc.legend()

plt.show()
