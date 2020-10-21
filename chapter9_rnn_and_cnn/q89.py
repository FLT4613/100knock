# https://pytorch.org/tutorials/intermediate/char_model_classification_tutorial.html

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# q83
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(device)
        self.optimizer = optimizer

    def train(self, train_dataset, test_dataset, epoch, batch_size):
        for _ in range(epoch):
            self.model.train()
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for x_batch, y_batch in tqdm(dataloader, total=len(dataloader)):
                loss, _ = self.model(input_ids=x_batch, labels=y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                train_x, train_y = train_dataset.tensors
                train_loss, train_logit = self.model(input_ids=train_x, labels=train_y)
                train_predict = torch.max(train_logit, 1)

                test_x, test_y = test_dataset.tensors
                test_loss, test_logit = self.model(input_ids=test_x, labels=test_y)
                test_predict = torch.max(test_logit, 1)

                print(f'Train Loss\t: {train_loss}')
                print(f'Train Accuracy\t: {torch.eq(train_y, train_predict.indices).sum().item() / len(train_y)}')
                print(f'Test Loss\t: {test_loss}')
                print(f'Test Accuracy\t: {torch.eq(test_y, test_predict.indices).sum().item() / len(test_y)}')


if __name__ == "__main__":

    # Dataset
    train = pd.read_table('files/train.txt', header=None)
    test = pd.read_table('files/test.txt',  header=None)

    # preprocess
    labels = ['b', 't', 'e', 'm']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_dataset(table):
        x = tokenizer(table.iloc[:, 0].to_list(), return_tensors="pt", padding=True)
        y = torch.tensor(table.iloc[:, 1].apply(labels.index).to_list())
        return TensorDataset(x.input_ids, y)

    batch_size = 10
    epoch = 10

    # Models
    bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
    optimizer = torch.optim.SGD(bert.parameters(), lr=0.01)

    trainer = Trainer(
        model=bert,
        optimizer=optimizer
    )

    train_dataset = get_dataset(train)
    test_dataset = get_dataset(train)

    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        epoch=epoch
    )
