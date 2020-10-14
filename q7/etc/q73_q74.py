import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

vec_x = torch.tensor(np.loadtxt('files/q7_train_x.txt'))
vec_y = torch.from_numpy(np.loadtxt('files/q7_train_y.txt')).long()
test_vec_x = torch.tensor(np.loadtxt('files/q7_test_x.txt'))
test_vec_y = pd.DataFrame(np.loadtxt('files/q7_test_y.txt'))

w = torch.randn(vec_x.shape[1], 4).double().requires_grad_(True)
learning_rate = 0.1
loss = nn.CrossEntropyLoss()

for _ in range(1000):
    y_pred = torch.matmul(vec_x, w)
    L = loss(y_pred, vec_y)
    L.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()

# q74
# mm == matmul
test_y_pred = F.softmax(torch.mm(test_vec_x, w), dim=1)
pred_labels = pd.DataFrame(test_y_pred).astype('float').apply(lambda x: x.idxmax(), axis=1)
e = pd.concat([pred_labels, test_vec_y], axis=1)
print(len(e[e.iloc[:, 0] == e.iloc[:, 1]]) / len(test_vec_y))
