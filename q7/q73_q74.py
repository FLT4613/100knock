from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


train_x = torch.tensor(np.loadtxt('files/q7_train_x.txt')).float()
train_y = torch.tensor(np.loadtxt('files/q7_train_y.txt')).long()
test_x = torch.tensor(np.loadtxt('files/q7_test_x.txt')).float()
test_y = torch.tensor(np.loadtxt('files/q7_test_y.txt')).long()

model = nn.Sequential(*[
    torch.nn.Linear(300, 4)
])

op = optim.SGD(model.parameters(), lr=0.1)
celoss = nn.CrossEntropyLoss()

epoch = 1000
losses = []
for i in tqdm(range(epoch)):
    predict = model(train_x)
    loss = celoss(predict, train_y)

    op.zero_grad()
    loss.backward()
    op.step()
    if i % 10 == 0:
        losses.append(loss.item())


# print(losses)
model.eval()
with torch.no_grad():
    predict = torch.max(model(test_x), 1)
    print(torch.eq(test_y, predict.indices).sum().item() / len(test_y))
