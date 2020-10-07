import pickle

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from pprint import pprint
from collections import OrderedDict
from q51 import load_data

with open('files/clf.pickle', 'rb') as f:
    clf = pickle.load(f)

train_x_vec = pd.read_table('files/train.feature.txt', header=None)
test_x_vec = pd.read_table('files/test.feature.txt', header=None)

_, train_y = load_data('files/train.txt')
_, test_y = load_data('files/test.txt')

with open('files/vocabulary.pickle', 'rb') as f:
    voc = pickle.load(f)


def print_score(feature, label):
    pred_y = clf.predict(feature)
    print(accuracy_score(label, pred_y))


# q55
print('--q54--')
print_score(train_x_vec, train_y)
print_score(test_x_vec, test_y)

# q55
print('--q55--')
pred_y = clf.predict(test_x_vec)
print(confusion_matrix(test_y, pred_y))

# q56
print('--q56--')
mapping = {
    'b': 'business',
    't': 'science and technology',
    'e': 'entertainment',
    'm': 'health'
}
print(
    classification_report(
        test_y, pred_y,
        target_names=[f'{mapping[x]}({x})' for x in clf.classes_]
    )
)

# q57
for coef, cl in zip(clf.coef_, clf.classes_):
    coef_voc = list(zip(coef, voc))
    top = sorted(coef_voc, key=lambda x: abs(x[0]), reverse=True)[:10]
    bottom = sorted(coef_voc, key=lambda x: abs(x[0]), reverse=False)[:10]
    pprint(
        OrderedDict([
            ('Category', mapping[cl]),
            ('Top 10', top),
            ('Bottom 10', bottom)
        ])
    )
