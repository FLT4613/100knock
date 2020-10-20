import pickle
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('files/googlenews_vectors')

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
