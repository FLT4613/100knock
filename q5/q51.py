from sklearn.feature_extraction.text import CountVectorizer
import re

import nltk
import pandas as pd
import pickle

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


cv_args = {
    'tokenizer': tokenize,
    'preprocessor': preprocess,
    'lowercase': True
}


if __name__ == "__main__":
    train_x, train_y = load_data('files/train.txt')
    valid_x, valid_y = load_data('files/valid.txt')
    test_x, test_y = load_data('files/test.txt')

    cv = CountVectorizer(**cv_args)
    pd.DataFrame(cv.fit_transform(train_x).toarray()).to_csv('files/train.feature.txt', sep='\t', index=False, header=None)
    pd.DataFrame(cv.transform(valid_x).toarray()).to_csv('files/valid.feature.txt', sep='\t', index=False, header=None)
    pd.DataFrame(cv.transform(test_x).toarray()).to_csv('files/test.feature.txt', sep='\t', index=False, header=None)
    with open('files/vocabulary.pickle', 'wb') as f:
        pickle.dump(cv.vocabulary_, f)
