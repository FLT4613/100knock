import spacy
import numpy as np
from pprint import pprint

nlp = spacy.load('files/googlenews_vectors')
d = {}
with open('files/questions-words.txt') as f:
    tmp = []
    for s in f.readlines():
        print(s)
        if s[0] == ':':
            if not tmp:
                d[s[1:].strip()] = np.array(tmp)
                tmp = []
                print(s)
        else:
            tmp.append(s.strip())

print(d)
# for title, dataset in d.items():
#     print(f'{title}')

#     exit()
