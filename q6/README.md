# 第7章: 単語ベクトル

## 準備

``` sh
curl -O https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
python -c "
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('files/GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('files/GoogleNews-vectors-negative300.txt', binary=False)"

gzip files/GoogleNews-vectors-negative300.txt
python -m spacy init-model en files/googlenews_vectors --vectors-loc files/GoogleNews-vectors-negative300.txt.gz
curl -o files/questions-words.txt http://download.tensorflow.org/data/questions-words.txt
curl -o files/wordsim353.zip http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip
unzip files/wordsim353.zip
```
