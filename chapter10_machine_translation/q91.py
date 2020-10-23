# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://huggingface.co/Helsinki-NLP/opus-mt-ja-en
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer)

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
                cfg = tokenizer.prepare_seq2seq_batch(x_batch, y_batch).to(device)
                loss, _, _ = self.model(**cfg)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
                train_losses = []
                train_accuracy = []
                for x_batch, y_batch in train_dataloader:
                    train_cfg = tokenizer.prepare_seq2seq_batch(x_batch, y_batch).to(device)
                    train_loss, train_logit, _ = self.model(**train_cfg)
                    train_losses.append(train_loss.item())
                    train_predict = torch.max(train_logit, 2)
                    for label, predict in zip(train_cfg.labels, train_predict.indices):
                        train_accuracy.append(torch.eq(label, predict).sum().item() / predict.shape[0])

                tune_dataloader = DataLoader(tune_dataset, batch_size=batch_size)
                tune_losses = []
                tune_accuracy = []
                for x_batch, y_batch in tune_dataloader:
                    tune_cfg = tokenizer.prepare_seq2seq_batch(x_batch, y_batch).to(device)
                    tune_loss, tune_logit, _ = self.model(**tune_cfg)
                    tune_losses.append(tune_loss.item())
                    tune_predict = torch.max(tune_logit, 2)
                    for label, predict in zip(tune_cfg.labels, tune_predict.indices):
                        tune_accuracy.append(torch.eq(label, predict).sum().item() / predict.shape[0])

                print(f'Train Loss\t: {np.array(train_losses).mean()}')
                print(f'Train Accuracy\t: {np.average(train_accuracy)}')
                print(f'Tune Loss\t: {np.array(tune_losses).mean()}')
                print(f'Tune Accuracy\t: {np.average(tune_accuracy)}')


if __name__ == "__main__":
    batch_size = 16
    epoch = 10

    src = None
    target = None

    def load(type_name):
        with open(f'files/kftt-data-1.0/data/orig/kyoto-{type_name}.ja', encoding='utf-8') as f:
            with open(f'files/kftt-data-1.0/data/orig/kyoto-{type_name}.en', encoding='utf-8') as g:
                return list(zip(f.readlines(), g.readlines()))

    train_dataset = load('train')
    tune_dataset = load('tune')

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
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
