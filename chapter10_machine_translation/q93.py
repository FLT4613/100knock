# https://cloud.google.com/translate/automl/docs/evaluate?hl=ja#interpretation

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk import bleu_score
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Translator:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    def translate(self, orig_str_list):
        with torch.no_grad():
            cfg = self.tokenizer.prepare_seq2seq_batch(orig_str_list).to(device)
            generated = self.model.generate(**cfg)
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    with open('files/kftt-data-1.0/data/orig/kyoto-test.ja', encoding='utf-8') as f:
        dataset_ja = f.readlines()
    with open('files/kftt-data-1.0/data/orig/kyoto-test.en', encoding='utf-8') as f:
        dataset_en = f.readlines()

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en", return_dict=True)
    model.load_state_dict(torch.load('files/model.pth'))
    translator = Translator(model, tokenizer)
    translated = translator.translate(dataset_ja)

    scores = np.array([bleu_score.sentence_bleu([ref.split()], hypo.split(), weights=(0.5, 0.5)) for ref, hypo in zip(dataset_en, translated)])
    print(scores.mean())
