import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

nlp = spacy.load('/home/ubuntu/anaconda3/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.3.1')

# 与えられた単語列に対して，ID番号の列を返す関数を実装せよ
try:
    with open('files/q80_output.pickle', 'rb') as f:
        x = pickle.load(f)

except FileNotFoundError:
    pass

# 学習データ中で2回以上出現する単語にID番号を付与せよ


def tokenizer(s):
    doc = nlp(s)
    return [token.lemma_ for token in doc if not token.is_stop and 'x' in token.shape_.lower()]


cv = CountVectorizer(
    tokenizer=tokenizer
)

train = pd.read_csv('files/train.txt', delimiter='\t', header=None).iloc[:, 0]

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform
feature = cv.fit_transform(train.values)
rank = sorted(
    zip(
        cv.get_feature_names(),
        feature.toarray().sum(axis=0),
    ),
    key=lambda x: x[1],
    reverse=True
)

with open('files/q80_output.pickle', 'wb') as f:
    pickle.dump({x[0]: i if x[1] > 1 else 0 for i, x in enumerate(rank, start=1)}, f)
