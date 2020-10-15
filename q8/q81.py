from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import torch.nn.functional as F

with open('files/q80_output.pickle', 'rb') as f:
    idlist = pickle.load(f)


with open('files/vocabulary.pickle', 'rb') as f:
    # from q52
    voc = pickle.load(f)


def to_idlist(s):
    # q80: 与えられた単語列に対して，ID番号の列を返す関数を実装せよ
    return [idlist.get(w, 0) for w in s]


d_w = 300
d_h = 50
L = 4

w_hx = torch.randn(d_w, d_h)
w_hh = torch.randn(d_h, d_h)
w_yh = torch.randn(L, d_h)

b = torch.randn(d_h)
b = torch.randn(d_h)


def emb(vec):
    return F.embedding(vec, w_hx)


def rnn(x, h):
    pass


def y(h_T, b_y):
    pass


title = "Marvel's Guardians of the Galaxy Test: Making Star-Lord Into an Actual Movie Star"
title_ids = to_idlist(title)
x = [
    torch.tensor([1 if i == title_id else 0 for i in range(1, len(voc))])
    for title_id in title_ids
]
print(emb(x[0]).shape)

exit()

print(b)
hat_y1 = F.softmax()
exit()
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
model = nn.Sequential(*[
    torch.nn.Linear(300, 4)
])
op = optim.SGD(model.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()
epoch = 10
model.train()

start = time()
