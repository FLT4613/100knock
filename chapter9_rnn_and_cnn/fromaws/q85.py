import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('files/q84_idlist.pickle', 'rb') as f:
    idlist = pickle.load(f)


with open('files/vocabulary_google.pickle', 'rb') as f:
    # from q52
    voc = pickle.load(f)


def to_idlist(s):
    # q80: 与えられた単語列に対して，ID番号の列を返す関数を実装せよ
    return [idlist.get(w, 0) for w in s]


# preprocess
def get_dataset(path):
    mapping = ['b', 't', 'e', 'm']
    table = pd.read_csv(path, delimiter='\t', header=None)
    table.iloc[:, 0] = table.iloc[:, 0].apply(to_idlist)
    table.iloc[:, 1] = table.iloc[:, 1].apply(lambda x: mapping.index(x))
    max_size = table.iloc[:, 0].apply(lambda x: len(x)).max()
    table.iloc[:, 0] = table.iloc[:, 0].apply(lambda x: [0] * (max_size - len(x)) + x)
    x = torch.tensor(table.iloc[:, 0]).to(device)
    y = torch.tensor(table.iloc[:, 1]).to(device)
    return TensorDataset(x, y)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(voc), input_size, padding_idx=0)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=3)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x = self.embedding(input)
        _, h = self.rnn(x)
        # 系列の最後の隠れ状態を使う
        y = self.softmax(self.linear(h[0]))
        return y


train_dataset = get_dataset('files/train.txt')
test_dataset = get_dataset('files/test.txt')

batch_size = 10
d_w = 300
d_h = 50
L = 4

rnn = Net(d_w, d_h, L).to(device)
loss_function = nn.NLLLoss()
op = optim.SGD(rnn.parameters(), lr=0.001)
losses = []
epoch = 100

for e in range(epoch):
    rnn.train()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for x_batch, y_batch in tqdm(dataloader, total=len(dataloader)):
        predict = rnn(x_batch)
        op.zero_grad()
        loss = loss_function(predict, y_batch)
        loss.backward()
        op.step()

    rnn.eval()
    with torch.no_grad():
        train_x, train_y = train_dataset.tensors
        train_result = rnn(train_x)
        train_predict = torch.max(train_result, 1)
        train_loss = loss_function(train_result, train_y)
        test_x, test_y = test_dataset.tensors
        test_result = rnn(test_x)
        test_predict = torch.max(test_result, 1)
        test_loss = loss_function(test_result, test_y)
        print(f'Train Loss\t: {train_loss}')
        print(f'Train Accuracy\t: {torch.eq(train_y, train_predict.indices).sum().item() / len(train_y)}')
        print(f'Test Loss\t: {test_loss}')
        print(f'Test Accuracy\t: {torch.eq(test_y, test_predict.indices).sum().item() / len(test_y)}')
