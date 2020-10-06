import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
whitelist = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']

df = pd.read_csv('files/newsCorpora.csv', delimiter="\t", header=None)
df = df[df.iloc[:, 3].isin(whitelist)].iloc[:, [1, 4]]
train, test = train_test_split(df, test_size=0.2)
valid = test[0::2]
test = test[1::2]

print('Train: ' + str(Counter(train.iloc[:, 1])))
print('Test: ' + str(Counter(test.iloc[:, 1])))

train.to_csv('files/train.txt', header=None, index=False, sep='\t')
valid.to_csv('files/valid.txt', header=None, index=False, sep='\t')
test.to_csv('files/test.txt', header=None, index=False, sep='\t')