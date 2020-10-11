import gensim
from tqdm import tqdm

model = gensim.models.KeyedVectors.load_word2vec_format('files/GoogleNews-vectors-negative300.bin', binary=True)

with open('files/questions-words.txt', encoding='utf-8') as f, open('files/output-q64.txt', 'w', encoding='utf-8') as g:
    for s in tqdm(f.readlines()):
        if s[0] == ':':
            g.write(s)
        else:
            word_list = s.strip().split()
            ret = model.most_similar(positive=[word_list[1], word_list[2]], negative=[word_list[0]])
            word_list.extend([str(x) for x in ret[0]])
            g.write(' '.join(word_list) + '\n')
