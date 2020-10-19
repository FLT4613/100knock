import pandas as pd
import numpy as np
import spacy


nlp = spacy.load('files/googlenews_vectors')


def get_x(s):
    text = s.split()
    return sum([nlp(x).vector for x in text]) / len(text)


def get_y(category):
    # (b = business, t = science and technology, e = entertainment, m = health)
    return ['b', 't', 'e', 'm'].index(category)


for name in ['train', 'valid', 'test']:
    table = pd.read_table(f'files/{name}.txt', header=None)
    x = np.vstack(table[0].apply(get_x))
    y = table[1].apply(get_y).to_numpy()
    np.savetxt(f'files/q7_{name}_x.txt', x)
    np.savetxt(f'files/q7_{name}_y.txt', y, fmt='%d')
