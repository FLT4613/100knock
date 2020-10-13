import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt


def pred(x, y):
    # mm == matmul
    pred = F.softmax(torch.mm(x, w), dim=1)
    labels = pd.DataFrame(pred).astype('float').apply(lambda x: x.idxmax(), axis=1)
    e = pd.concat([labels, y], axis=1)
    return len(e[e.iloc[:, 0] == e.iloc[:, 1]]) / len(y)


train_x = torch.tensor(np.loadtxt('files/q7_train_x.txt'))
train_y = pd.DataFrame(np.loadtxt('files/q7_train_y.txt'))

valid_x = torch.tensor(np.loadtxt('files/q7_valid_x.txt'))
valid_y = pd.DataFrame(np.loadtxt('files/q7_valid_y.txt'))

test_x = torch.tensor(np.loadtxt('files/q7_test_x.txt'))
test_y = pd.DataFrame(np.loadtxt('files/q7_test_y.txt'))

w = torch.randn(train_x.shape[1], 4).double().requires_grad_(True)
learning_rate = 0.1
loss = nn.CrossEntropyLoss()

ax_loss = plt.subplot(2, 1, 2)
ax_acc = plt.subplot(2, 1, 1)
datapoints = pd.DataFrame()

for _ in range(1000):
    train_pred = torch.matmul(train_x, w)
    valid_pred = torch.matmul(valid_x, w)

    L_t = loss(train_pred, train_y)
    L_v = loss(valid_pred, valid_y)

    L_t.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()

    datapoints.add(pd.DataFrame([
        L_t, L_v,
        pred(train_x, train_y), pred(valid_x, valid_y)
    ]).astype('float'))


# q74
print(pred(test_x, test_y))

# test_y_pred = F.softmax(torch.mm(test_x, w), dim=1)
# pred_labels = pd.DataFrame(test_y_pred).astype('float').apply(lambda x: x.idxmax(), axis=1)
# e = pd.concat([pred_labels, test_y], axis=1)
# print(len(e[e.iloc[:, 0] == e.iloc[:, 1]]) / len(test_y))

ax_loss.plot(
    range(1000),
    datapoints.iloc[:.0],
    label='train',
    marker='o'
)
ax_loss.plot(
    range(1000),
    datapoints.iloc[:.1],
    label='valid',
    marker='o'
)
ax_acc.plot(
    range(1000),
    datapoints.iloc[:.2],
    label='train',
    marker='o'
)
ax_acc.plot(
    range(1000),
    datapoints.iloc[:.2],
    label='valid',
    marker='o'
)


plt.show()
