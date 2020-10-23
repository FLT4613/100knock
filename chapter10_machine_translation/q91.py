# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://huggingface.co/Helsinki-NLP/opus-mt-ja-en
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import spacy
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# q83
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(device)
        self.optimizer = optimizer

    def train(self, train_dataset, tune_dataset, epoch, batch_size):
        for _ in range(epoch):
            self.model.train()
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for x_batch, y_batch in tqdm(dataloader, total=len(dataloader)):
                cfg = tokenizer.prepare_seq2seq_batch(x_batch, y_batch)
                # print(cfg)
                loss = self.model(**cfg)
                print(loss)
                exit()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                train_x, train_y = train_dataset.tensors
                cfg = tokenizer.prepare_seq2seq_batch(train_x, train_y)
                train_loss, train_logit = self.model(**cfg)
                train_predict = torch.max(train_logit, 1)

                tune_x, tune_y = tune_dataset.tensors
                cfg = tokenizer.prepare_seq2seq_batch(tune_x, tune_y)
                tune_loss, tune_logit = self.model(**cfg)
                tune_predict = torch.max(tune_logit, 1)

                print(f'Train Loss\t: {train_loss}')
                print(f'Train Accuracy\t: {torch.eq(train_y, train_predict.indices).sum().item() / len(train_y)}')
                print(f'Tune Loss\t: {tune_loss}')
                print(f'Tune Accuracy\t: {torch.eq(tune_y, tune_predict.indices).sum().item() / len(tune_y)}')


if __name__ == "__main__":
    batch_size = 10
    epoch = 10

    src = None
    target = None

    def load(type_name):
        with open(f'files/kftt-data-1.0/data/tok/kyoto-{type_name}.ja') as f:
            with open(f'files/kftt-data-1.0/data/tok/kyoto-{type_name}.en') as g:
                return list(zip(f.readlines(), g.readlines()))

    train_dataset = load('train')
    tune_dataset = load('tune')

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model=model,
        optimizer=optimizer
    )

    trainer.train(
        train_dataset=train_dataset,
        tune_dataset=tune_dataset,
        batch_size=batch_size,
        epoch=epoch
    )
