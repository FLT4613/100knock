# https://pytorch.org/tutorials/intermediate/char_model_classification_tutorial.html

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import spacy

# q83
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlp):
        super(RNN, self).__init__()
        self.model = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.embedding = nn.Embedding(nlp.vocab.vectors.shape[0], input_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))

    def forward(self, input):
        x = self.embedding(input)
        _, h = self.model(x)
        y = self.softmax(self.linear(h.squeeze(0)))
        return y


class Preprocessor:
    def __init__(self, nlp):
        self.nlp = nlp

    def to_idlist(self, s):
        # q84
        return [self.nlp.vocab.vectors.key2row.get(w.norm, 0) for w in self.nlp(s)]

    def get_dataset(self, path):
        mapping = ['b', 't', 'e', 'm']
        table = pd.read_csv(path, delimiter='\t', header=None)
        table.iloc[:, 0] = table.iloc[:, 0].apply(self.to_idlist)
        table.iloc[:, 1] = table.iloc[:, 1].apply(mapping.index)
        max_size = table.iloc[:, 0].apply(len).max()
        table.iloc[:, 0] = table.iloc[:, 0].apply(lambda x: [0] * (max_size - len(x)) + x)
        x = torch.tensor(table.iloc[:, 0]).to(device)
        y = torch.tensor(table.iloc[:, 1]).to(device)
        return TensorDataset(x, y)


class Trainer:
    def __init__(self, model, loss_function, optimizer):
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, train_dataset, test_dataset, epoch, batch_size):
        for _ in range(epoch):
            self.model.train()
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for x_batch, y_batch in tqdm(dataloader, total=len(dataloader)):
                predict = self.model(x_batch)
                self.optimizer.zero_grad()
                loss = self.loss_function(predict, y_batch)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                train_x, train_y = train_dataset.tensors
                train_result = self.model(train_x)
                train_predict = torch.max(train_result, 1)
                train_loss = self.loss_function(train_result, train_y)
                test_x, test_y = test_dataset.tensors
                test_result = self.model(test_x)
                test_predict = torch.max(test_result, 1)
                test_loss = self.loss_function(test_result, test_y)
                print(f'Train Loss\t: {train_loss}')
                print(f'Train Accuracy\t: {torch.eq(train_y, train_predict.indices).sum().item() / len(train_y)}')
                print(f'Test Loss\t: {test_loss}')
                print(f'Test Accuracy\t: {torch.eq(test_y, test_predict.indices).sum().item() / len(test_y)}')


if __name__ == "__main__":
    d_w = 300
    d_h = 128
    L = 4

    # Dataset
    # Ref: chapter7_word_vector/README.md
    nlp = spacy.load('files/googlenews_vectors')

    preprocessor = Preprocessor(nlp)

    train_dataset = preprocessor.get_dataset('files/train.txt')
    test_dataset = preprocessor.get_dataset('files/test.txt')

    batch_size = 1
    epoch = 10

    # Models
    rnn = RNN(d_w, d_h, L, nlp)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.01)

    trainer = Trainer(
        model=rnn,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        epoch=epoch
    )
