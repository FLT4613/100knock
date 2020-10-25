# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://huggingface.co/Helsinki-NLP/opus-mt-ja-en
# https://huggingface.co/transformers/model_doc/marian.html#marianmt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, AdamW)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

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
                train_loss, train_accuracy = self._eval(train_dataloader)

                tune_dataloader = DataLoader(tune_dataset, batch_size=batch_size)
                tune_loss, tune_accuracy = self._eval(tune_dataloader)

                print(f'Train Loss\t: {train_loss}')
                print(f'Train Accuracy\t: {train_accuracy}')
                print(f'Tune Loss\t: {tune_loss}')
                print(f'Tune Accuracy\t: {tune_accuracy}')

        torch.save(model.to('cpu').state_dict(), 'files/model.pth')
        model.to(device)

    def _eval(self, dataloader):
        losses = torch.tensor(0.)
        accuracies = torch.tensor(0.)
        for x_batch, y_batch in dataloader:
            cfg = tokenizer.prepare_seq2seq_batch(x_batch, y_batch).to(device)
            loss, _, _ = self.model(**cfg)
            losses += loss
            predict = self.model.generate(**cfg)
            predict_str = self.tokenizer.batch_decode(predict, skip_special_tokens=True)
            accuracies += sum(x.strip() == y for x, y in zip(y_batch, predict_str))
        return (losses / len(dataloader)).item(), (accuracies / len(dataloader)).item()


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

    optimizer = AdamW(model.parameters(), lr=0.001)
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
