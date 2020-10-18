import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul, tanh

# https://pytorch.org/docs/stable/generated/torch.set_printoptions.html#torch-set-printoptions
# 指数表記を無効にする
torch.set_printoptions(sci_mode=False)

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

w_hx = torch.randn(d_h, d_w)
w_hh = torch.randn(d_h, d_h)
w_yh = torch.randn(L, d_h)

b_h = torch.randn(d_h)
b_y = torch.randn(L)

h_T = torch.tensor([])


def rnn(x, h):
    return tanh(matmul(w_hx, x.T) + matmul(w_hh, h))


def y(h_T):
    return F.softmax(matmul(w_yh, h_T), dim=0)


# preprocess
train = pd.read_csv('files/train.txt', delimiter='\t', header=None).iloc[:, 0]
id_list = sorted(train.apply(to_idlist).tolist(), key=lambda x: len(x), reverse=True)
max_size = len(id_list[0])
emb = nn.Embedding(len(voc), d_w, padding_idx=0)
x = torch.tensor([(x + [0] * (max_size - len(x))) for x in id_list])

# # fitting
h_t = torch.zeros(d_h, len(train))

for i in range(max_size):
    x_t = x[:, i]
    h_t = rnn(emb(x_t), h_t)

print(y(h_t).T)
