# 第7章: 単語ベクトル

## 準備

``` sh
python -c "
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('files/GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('files/GoogleNews-vectors-negative300.txt', binary=False)"

gzip files/GoogleNews-vectors-negative300.txt
python -m spacy init-model en files/googlenews_vectors --vectors-loc files/GoogleNews-vectors-negative300.txt.gz
curl -o files/questions-words.txt http://download.tensorflow.org/data/questions-words.txt
```

## 備考

* Google Newsデータセットは1.5GB程度あるので注意