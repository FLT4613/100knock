import pandas as pd
from sklearn.linear_model import LogisticRegression
from q51 import load_data
import pickle


def create_model(train_vec, target_vec, **lr_kwargs):
    clf = LogisticRegression(solver='liblinear', **lr_kwargs)
    return clf.fit(train_vec, target_vec)


if __name__ == '__main__':
    _, train_y = load_data('files/train.txt')
    train_x_vec = pd.read_csv('files/train.feature.txt', delimiter='\t', header=None)
    model = create_model(train_x_vec, train_y)
    with open('files/clf.pickle', 'wb') as f:
        pickle.dump(model, f)
