import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def pred(x, y):
    # mm == matmul
    pred = F.softmax(torch.mm(x, w), dim=1)
    labels = pd.DataFrame(pred).astype('float').apply(lambda x: x.idxmax(), axis=1)
    return sum(1 for (l, r) in zip(labels, y) if l == r) / len(y)


train_x = torch.tensor(np.loadtxt('files/q7_train_x.txt'))
train_y = torch.tensor(np.loadtxt('files/q7_train_y.txt')).long()

valid_x = torch.tensor(np.loadtxt('files/q7_valid_x.txt'))
valid_y = torch.tensor(np.loadtxt('files/q7_valid_y.txt')).long()


w = torch.randn(train_x.shape[1], 4).double().requires_grad_(True)
learning_rate = 0.1
loss = nn.CrossEntropyLoss()
epoch = 1000

datapoints = []
ax_loss = plt.subplot(2, 1, 2)
ax_acc = plt.subplot(2, 1, 1)

for _ in tqdm(range(epoch)):
    train_pred = torch.matmul(train_x, w)
    valid_pred = torch.matmul(valid_x, w)

    L_t = loss(train_pred, train_y)
    L_v = loss(valid_pred, valid_y)

    L_t.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()

    datapoints.append([
        L_t.item(), L_v.item(),
        pred(train_x, train_y), pred(valid_x, valid_y)
    ])

df = pd.DataFrame(datapoints)
ax_loss.set_title("Loss")
ax_loss.plot(
    range(epoch),
    df.iloc[:, 0],
    label='train',
    marker='o'
)
ax_loss.plot(
    range(epoch),
    df.iloc[:, 1],
    label='valid',
    marker='x'
)
ax_acc.set_title("Accuracy")
ax_acc.plot(
    range(epoch),
    df.iloc[:, 2],
    label='train',
    marker='x'
)
ax_acc.plot(
    range(epoch),
    df.iloc[:, 3],
    label='valid',
    marker='x'
)

ax_loss.legend()
ax_acc.legend()
plt.show()
