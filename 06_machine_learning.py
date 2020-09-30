
import numpy as np
from random import shuffle


def preprocess():
    publishers = [
        "Reuters",
        "Huffington Post",
        "Businessweek",
        "Contactmusic.com",
        "Daily Mail"
    ]
    dataset = []
    with open('NewsAggregatorDataset/newsCorpora.csv') as f:
        for row in f.readlines():
            data = row.split('\t')
            if data[3] in publishers:
                dataset.append('\t'.join([data[1], data[4]]))
    shuffle(dataset)
    train = dataset[0:int(len(dataset) * 8 / 10)]
    valid = dataset[int(len(dataset) * 8 / 10):int(len(dataset) * 9 / 10)]
    test = dataset[int(len(dataset) * 9 / 10):]
    with open('train.txt', 'w') as f:
        f.write('\n'.join(train))
    with open('valid.txt', 'w') as f:
        f.write('\n'.join(valid))
    with open('test.txt', 'w') as f:
        f.write('\n'.join(test))


# preprocess()

theta = np.random.rand()

# ①データの準備
# ②シグモイド関数の実装


def sigmoid(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

# ③パラメータ更新式の実装
# ④パラメータの更新
# ⑤結果の確認
