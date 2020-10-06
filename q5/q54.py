import pickle

import pandas as pd
from sklearn.metrics import accuracy_score

from q51 import load_data


with open('files/clf.pickle', 'rb') as f:
    clf = pickle.load(f)

train_x_vec = pd.read_table('files/train.feature.txt', header=None)
test_x_vec = pd.read_table('files/test.feature.txt', header=None)

_, train_y = load_data('files/train.txt')
_, test_y = load_data('files/test.txt')


def print_score(feature, label):
    pred_y = clf.predict(feature)
    print(accuracy_score(label, pred_y))


print_score(train_x_vec, train_y)
print_score(test_x_vec, test_y)
