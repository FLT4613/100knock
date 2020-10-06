import re

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

stopwords = nltk.corpus.stopwords.words('english')


def load_data(path):
    ret = pd.read_csv(path, delimiter='\t', header=None)
    return [
        ret.iloc[:, 0],
        ret.iloc[:, 1],
    ]


def preprocess(s):
    s = re.sub(r"[\"\#]", "", s)
    return re.sub(r'\d+', '0', s)


def tokenize(s):
    return [w for w in s.split() if w not in stopwords]


train_x, train_y = load_data('train.txt')
valid_x, valid_y = load_data('valid.txt')
test_x, test_y = load_data('test.txt')

cv = CountVectorizer(lowercase=True, preprocessor=preprocess, tokenizer=tokenize)

pd.DataFrame(cv.fit_transform(train_x).toarray()).to_csv('train.feature.txt', sep='\t', index=False, header=None)
pd.DataFrame(cv.transform(valid_x).toarray()).to_csv('valid.feature.txt', sep='\t', index=False, header=None)
pd.DataFrame(cv.transform(test_x).toarray()).to_csv('test.feature.txt', sep='\t', index=False, header=None)
