import spacy
import pandas as pd
from pprint import pprint
import numpy as np
nlp = spacy.load('files/googlenews_vectors')
d = {}
with open('files/questions-words.txt') as f:
    tmp = []
    for s in f.readlines():
        if s[0] == ':':
            if tmp:
                d[s] = pd.DataFrame(tmp).applymap(nlp)
                tmp.clear()
        else:
            tmp.append(s.strip().split())


def most_similar(vec):
    ms = nlp.vocab.vectors.most_similar(
        np.asarray([vec]),
        n=1
    )
    return [nlp.vocab.strings[w] for w in ms[0][0]][0]


def to_vector(nlp):
    return nlp.vocab.vectors.data


for title, dataset in d.items():
    t = dataset.applymap(to_vector)
    vec = t[2] - t[1] + t[3]
    data = pd.merge([dataset.applymap(lambda x: x.text), vec.applymap(most_similar)])
    pd.concat([title], data).to_csv('files/output-q64.txt', mode='a', header=None, index=None, sep=' ')
