# Ref:
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

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

try:
    checkpoint = torch.load('files/q7_model.pth')
except FileNotFoundError:
    checkpoint = {}


def get_accuracy_score(pred, y):
    return torch.eq(torch.max(pred, 1).indices, y).sum().item() / len(y)


model = nn.Sequential(*[torch.nn.Linear(300, 4)])
op = optim.SGD(model.parameters(), lr=0.1)

if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    op.load_state_dict(checkpoint['optimizer_state_dict'])

loss = checkpoint.get('loss', nn.CrossEntropyLoss())
epoch = checkpoint.get('epoch', 0)
datapoints = checkpoint.get('datapoints', [])
next_epoch = epoch + 100

for i in tqdm(range(epoch, next_epoch)):
    model.train()
    train_pred = model(train_x)
    train_loss = loss(train_pred, train_y)
    op.zero_grad()
    train_loss.backward()
    op.step()

    model.eval()
    with torch.no_grad():
        valid_pred = model(valid_x)
        valid_loss = loss(valid_pred, valid_y)

        datapoints.append([
            train_loss.item(),
            valid_loss.item(),
            get_accuracy_score(train_pred, train_y),
            get_accuracy_score(valid_pred, valid_y)
        ])

    torch.save({
        'epoch': i+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': op.state_dict(),
        'loss': loss,
        'datapoints': datapoints
    }, 'files/q7_model.pth')

df = pd.DataFrame(datapoints)

ax_loss = plt.subplot(2, 1, 2)
ax_loss.set_title("Loss")
ax_loss.plot(range(next_epoch), df.iloc[:, 0], label='train',)
ax_loss.plot(range(next_epoch), df.iloc[:, 1], label='valid',)
ax_loss.legend()

ax_acc = plt.subplot(2, 1, 1)
ax_acc.set_title("Accuracy")
ax_acc.plot(range(next_epoch), df.iloc[:, 2], label='train',)
ax_acc.plot(range(next_epoch), df.iloc[:, 3], label='valid',)
ax_acc.legend()

plt.show()
