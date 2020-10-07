import pandas as pd


def load_data(path):
    ret = pd.read_csv(path, delimiter='\t', header=None)
    return [
        ret.iloc[:, 0],
        ret.iloc[:, 1],
    ]
