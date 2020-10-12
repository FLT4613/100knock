import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

vec_x = torch.from_numpy(np.loadtxt('files/q7_train_x.txt'))
table_y = torch.from_numpy(np.loadtxt('files/q7_train_y.txt'))

w = torch.randn(vec_x.shape[1], 4).double()

print(vec_x[0])
print(w.shape)

# print(F.softmax(vec_x[0] * w.T, dim=1).shape)
# print(F.softmax(vec_x[0] * w.T, dim=1).shape)
print(F.softmax(torch.matmul(vec_x[0], w)))
print(F.softmax(torch.matmul(vec_x[0:4], w)))
