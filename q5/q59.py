import pandas as pd
from sklearn.linear_model import LogisticRegression
from q51 import load_data
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

_, train_y = load_data('files/train.txt')
_, test_y = load_data('files/train.txt')
train_x_vec = pd.read_csv('files/train.feature.txt', delimiter='\t', header=None)
test_x_vec = pd.read_csv('files/test.feature.txt', delimiter='\t', header=None)

lr_gscv = GridSearchCV(
    LogisticRegression(),
    {
        'C': [10**(-c) for c in range(-2, 4)],
        'random_state': [42],
        'penalty': ['l1', 'l2']
    },
    cv=4,
    verbose=2
)
lr_gscv.fit(train_x_vec, train_y)
pred_y = lr_gscv.best_estimator_.predict(test_x_vec)
print(accuracy_score(test_y, pred_y))
