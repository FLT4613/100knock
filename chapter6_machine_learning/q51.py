from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import pickle
import spacy

nlp = spacy.load('en_core_web_sm')


def load_data(path):
    ret = pd.read_csv(path, delimiter='\t', header=None)
    return [
        ret.iloc[:, 0],
        ret.iloc[:, 1],
    ]


def tokenizer(s):

    doc = nlp(s)
    return [token.lemma_ for token in doc]


cv_args = {
    'tokenizer': tokenizer,
}


if __name__ == '__main__':
    train_x, train_y = load_data('files/train.txt')
    valid_x, valid_y = load_data('files/valid.txt')
    test_x, test_y = load_data('files/test.txt')

    cv = CountVectorizer(**cv_args)
    pd.DataFrame(cv.fit_transform(train_x).toarray()).to_csv('files/train.feature.txt', sep='\t', index=False, header=None)
    pd.DataFrame(cv.transform(valid_x).toarray()).to_csv('files/valid.feature.txt', sep='\t', index=False, header=None)
    pd.DataFrame(cv.transform(test_x).toarray()).to_csv('files/test.feature.txt', sep='\t', index=False, header=None)
    with open('files/vocabulary.pickle', 'wb') as f:
        pickle.dump(cv.vocabulary_, f)
