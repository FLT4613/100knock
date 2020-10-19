import spacy
import numpy as np


nlp = spacy.load('files/googlenews_vectors')
print('---q60---')
print(nlp('United_States').vector)

print('---q61---')
print(nlp('United_States').similarity(nlp('U.S.')))

print('---q62---')
ms = nlp.vocab.vectors.most_similar(
    np.asarray([nlp('United_States').vector]),
    n=11
)
# https://spacy.io/api/vectors#most_similar
# > The most similar entries as a (keys, best_rows, scores) tuple.
print([nlp.vocab.strings[w] for w in ms[0][0]][1:])


print('---q63---')
vec = nlp('Spain').vector - nlp('Madrid').vector + nlp('Athens').vector
ms = nlp.vocab.vectors.most_similar(
    np.asarray([vec]),
    n=11
)
# https://spacy.io/api/vectors#most_similar
# > The most similar entries as a (keys, best_rows, scores) tuple.
print([nlp.vocab.strings[w] for w in ms[0][0]][1:])
