import pandas as pd
from q51 import cv_args
from sys import argv
import pickle
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    if not (argv[1] and isinstance(argv[1], str)):
        exit(1)
    with open('files/clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    with open('files/vocabulary.pickle', 'rb') as f:
        voc = pickle.load(f)
    cv = CountVectorizer(
        vocabulary=voc,
        **cv_args
    )
    mapping = {
        'b': 'business',
        't': 'science and technology',
        'e': 'entertainment',
        'm': 'health'
    }
    print(
        pd.DataFrame(
            clf.predict_proba(cv.transform([argv[1]])),
            columns=[f'{mapping[x]}({x})' for x in clf.classes_],
        ).to_string(
            index=False
        )
    )
