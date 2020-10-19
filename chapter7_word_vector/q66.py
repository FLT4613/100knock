import pandas as pd
import gensim


model = gensim.models.KeyedVectors.load_word2vec_format('files/GoogleNews-vectors-negative300.bin', binary=True)

table = pd.read_csv('files/combined.csv').sort_values(by=["Human (mean)"], ascending=False).reset_index(drop=True)
table.index += 1
table.reset_index(inplace=True)

wv_rank = pd.concat(
    [
        table.iloc[:, [1, 2]],
        table.apply(lambda x: model.similarity(x[1], x[2]), axis=1).rename("Model Output")
    ],
    axis=1
).sort_values(by=["Model Output"], ascending=False).reset_index(drop=True)
wv_rank.index += 1
wv_rank.reset_index(inplace=True)

ranking_table = table.merge(wv_rank, on=['Word 1', 'Word 2'], suffixes=['_human', '_model'])

summation = ranking_table.apply(lambda x: (x[0] - x[4]) ** 2, axis=1).sum()
n = len(ranking_table)
rho = 1 - (6 * summation) / (n * (n**2-1))
print(rho)
