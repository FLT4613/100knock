import re

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from q51 import load_data


train_x, train_y = load_data('train.txt')
train_x_vec = pd.read_csv('train.feature.txt', delimiter='\t', header=None)
clf = LogisticRegression(solver='liblinear')
clf.fit(train_x_vec, train_y)
