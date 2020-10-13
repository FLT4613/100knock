import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

vec_x = torch.tensor(np.loadtxt('files/q7_train_x.txt'))
vec_y = torch.from_numpy(np.loadtxt('files/q7_train_y.txt')).long()
w = torch.randn(vec_x.shape[1], 4).double().requires_grad_(True)
w_Y = w.clone().detach().requires_grad_(True)

# q71
hat_y1 = F.softmax(torch.matmul(vec_x[:1], w), dim=1)
hat_Y = F.softmax(torch.matmul(vec_x[:4], w_Y), dim=1)

print(hat_y1)
print(hat_Y)

# q72
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# CrossEntropyLossに含まれているため、softmaxは使わない
# https://discuss.pytorch.org/t/why-does-crossentropyloss-include-the-softmax-function/4420

loss = nn.CrossEntropyLoss()

p1 = torch.matmul(vec_x[:1], w)
p1_4 = torch.matmul(vec_x[:4], w_Y)
l1 = loss(p1, vec_y[:1])
l1_4 = loss(p1_4, vec_y[:4])

print(l1)
print(l1_4)

l1.backward()
l1_4.backward()
print(w.grad)
print(w_Y.grad)
