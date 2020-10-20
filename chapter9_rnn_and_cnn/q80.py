import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

nlp = spacy.load('en_core_web_sm')


def tokenizer(s):
    doc = nlp(s)
    return [token.lemma_ for token in doc if not token.is_stop and 'x' in token.shape_.lower()]


cv = CountVectorizer(
    tokenizer=tokenizer
)

train = pd.read_csv('files/train.txt', delimiter='\t', header=None).iloc[:, 0]

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
feature = cv.fit_transform(train.values)
rank = sorted(
    zip(
        cv.get_feature_names(),
        feature.toarray().sum(axis=0),
    ),
    key=lambda x: x[1],
    reverse=True
)

with open('files/q80_output.pickle', 'wb') as f:
    pickle.dump({x[0]: i if x[1] > 1 else 0 for i, x in enumerate(rank, start=1)}, f)
