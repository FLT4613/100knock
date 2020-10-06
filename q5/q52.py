import pandas as pd
from sklearn.linear_model import LogisticRegression
from q51 import load_data
import pickle

train_x, train_y = load_data('files/train.txt')
test_x, test_y = load_data('files/test.txt')
train_x_vec = pd.read_csv('files/train.feature.txt', delimiter='\t', header=None)
test_x_vec = pd.read_csv('files/test.feature.txt', delimiter='\t', header=None)
clf = LogisticRegression(solver='liblinear')
clf.fit(train_x_vec, train_y)
with open('files/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
