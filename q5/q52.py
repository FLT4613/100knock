import pandas as pd
from sklearn.linear_model import LogisticRegression
from util import load_data


train_x, train_y = load_data('files/train.txt')
train_x_vec = pd.read_csv('files/train.feature.txt', delimiter='\t', header=None)
clf = LogisticRegression(solver='liblinear')
clf.fit(train_x_vec, train_y)
