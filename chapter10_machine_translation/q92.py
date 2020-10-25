import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Translator:
    # q92: 91で学習したニューラル機械翻訳モデルを用い，与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    def translate(self, orig_str_list):
        with torch.no_grad():
            cfg = self.tokenizer.prepare_seq2seq_batch(orig_str_list).to(device)
            generated = self.model.generate(**cfg)
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    with open('files/kftt-data-1.0/data/orig/kyoto-dev.ja', encoding='utf-8') as f:
        dataset_ja = f.readlines()
    with open('files/kftt-data-1.0/data/orig/kyoto-dev.en', encoding='utf-8') as f:
        dataset_en = f.readlines()

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en", return_dict=True)
    model.load_state_dict(torch.load('files/model.pth'))
    translator = Translator(model, tokenizer)
    translated = translator.translate(dataset_ja)

    for num, (original, expected, result) in enumerate(zip(dataset_ja, dataset_en, translated), start=1):
        print(f'--- #{num} ---')
        print(f'日: {original}英: {expected}訳: {result}')
