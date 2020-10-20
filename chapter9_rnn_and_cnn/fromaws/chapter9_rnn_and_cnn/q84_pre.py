import numpy as np
import torch.nn as nn
import torch
import pickle
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('files/googlenews_vectors')
# print(nlp("Pen").vector)
# nlp = spacy.load('en_core_web_md')

n_vocab, vocab_dim = nlp.vocab.vectors.shape
emb = nn.Embedding(n_vocab, vocab_dim)

# Load pretrained embeddings
emb.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))

# --- Equivilent test for Spacy.nlp and torch.embeddings ---
test_vocab = ['apple', 'bird', 'cat', 'dog', 'egg', 'e12dsafdsf1']

# dict for converting vocab to row index for word vector matrix
key2row = nlp.vocab.vectors.key2row


for v in test_vocab:
    vocab_id = nlp.vocab.strings[v]
    spacy_vec = nlp.vocab[v].vector
    row = key2row.get(vocab_id, None)
    if row is None:
        print('{} is oov'.format(v))
        continue
    print(vocab_id)
    vocab_row = torch.tensor(row, dtype=torch.long)
    print(vocab_row)
    embed_vec = emb(vocab_row)
    print(np.allclose(spacy_vec, embed_vec.detach().numpy()))

exit()
# From q52


def load_data(path):
    ret = pd.read_csv(path, delimiter='\t', header=None)
    return [
        ret.iloc[:, 0],
        ret.iloc[:, 1],
    ]


def tokenizer(s):
    doc = nlp(s)
    return [token.lemma_ for token in doc]


train = pd.read_csv('files/train.txt', delimiter='\t', header=None).iloc[:, 0]
cv = CountVectorizer(tokenizer=tokenizer)
feature = cv.fit_transform(train)

rank = sorted(
    zip(
        cv.get_feature_names(),
        feature.toarray().sum(axis=0),
    ),
    key=lambda x: x[1],
    reverse=True
)

with open('files/q84_idlist.pickle', 'wb') as f:
    pickle.dump({x[0]: i if x[1] > 1 else 0 for i, x in enumerate(rank, start=1)}, f)

with open('files/vocabulary_google.pickle', 'wb') as f:
    pickle.dump(cv.vocabulary_, f)
