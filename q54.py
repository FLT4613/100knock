import re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_y_pred = clf.predict(train_x_vec)
y_pred = clf.predict(test_x_vec)
score1 = accuracy_score(train_y, train_y_pred)
score = accuracy_score(test_y, y_pred)

print(score)
print(score1)
