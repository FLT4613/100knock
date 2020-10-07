import matplotlib.pyplot as plt
import pandas as pd

from q51 import load_data
from q52 import create_model

train_x_vec = pd.read_table('files/train.feature.txt', header=None)
valid_x_vec = pd.read_table('files/valid.feature.txt', header=None)
test_x_vec = pd.read_table('files/test.feature.txt', header=None)

_, train_y = load_data('files/train.txt')
_, valid_y = load_data('files/valid.txt')
_, test_y = load_data('files/test.txt')

for index, penalty in enumerate(['l1', 'l2'], start=1):
    models = [
        create_model(train_x_vec, train_y, C=10**(-c), random_state=42, penalty=penalty)
        for c in range(-2, 6)
    ]
    plt.subplot(2, 1, index)
    plt.xscale('log')
    plt.title(f'{penalty.upper()}')
    plt.plot(
        [10**(-c) for c in range(-2, 6)],
        [model.score(train_x_vec, train_y) for model in models],
        label='train',
        marker='o'
    )
    plt.plot(
        [10**(-c) for c in range(-2, 6)],
        [model.score(valid_x_vec, valid_y) for model in models],
        label='valid',
        marker='o'
    )
    plt.plot(
        [10**(-c) for c in range(-2, 6)],
        [model.score(test_x_vec, test_y) for model in models],
        label='test',
        marker='o'
    )
    plt.ylim([0, 1])
    plt.legend()

plt.show()
