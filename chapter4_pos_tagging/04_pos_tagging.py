from itertools import groupby
import matplotlib.pyplot as plt
from collections import Counter
import MeCab
from pprint import pprint

from matplotlib import rcParams
rcParams['font.family'] = 'Ricty Diminished'


def q30():
    with open('neko.txt.mecab') as f:
        words = f.read().split('\n')
    lst = []
    for w in words:
        if w == 'EOS' or not w:
            continue
        surface, rest = w.split('\t')
        if not surface:
            continue
        sp = rest.split(',')
        pos, pos1, base = sp[0], sp[1], sp[-3]
        lst.append({
            'surface': surface,
            'base': base,
            'pos': pos,
            'pos1': pos1
        })

    return lst


data = q30()


def q31():
    return [x['surface'] for x in data if x['pos'] == '動詞']


def q32():
    return [x['base'] for x in data if x['pos'] == '動詞']


def q33():
    lst = []
    for i in range(len(data)-2):
        if data[i]['pos'] == '名詞' and data[i+1]['surface'] == 'の' and data[i+2]['pos'] == '名詞':
            lst.append(data[i]['surface'] + data[i+1]
                       ['surface'] + data[i+2]['surface'])
    return lst
    return [x for x in data if x['surface'] == 'の']


def q34():
    lst = []
    tmp = ""
    streak = False
    for i in range(len(data)):
        if data[i]['pos'] == '名詞':
            tmp += data[i]['surface']
        else:
            if tmp and len(tmp):
                lst.append(tmp)
            tmp = ""

    return lst


def q35():
    return sorted(
        [(k, v) for k, v in Counter([x['surface']
                                     for x in data if x['pos'] != '記号']).items()],
        key=lambda x: x[1], reverse=True
    )


def q36():
    res = q35()[:10]
    x = [d[0] for d in res]
    y = [d[1] for d in res]
    plt.bar(x, y)  # Plot some data on the axes.
    plt.show()


def q37():
    tmp = []
    rank = []
    for i in range(len(data)):
        if data[i]['surface'] == '。':
            if '猫' in tmp:
                rank += [t for t in tmp if t != '猫']
            tmp.clear()
        else:
            tmp.append(data[i]['surface'])
    res = sorted(
        [(k, v) for k, v in Counter(rank).items()], key=lambda x: x[1], reverse=True
    )[:30]
    x = [d[0] for d in res]
    y = [d[1] for d in res]
    plt.bar(x, y)
    plt.show()


def q38_q39():
    rank = sorted(Counter([d['surface'] for d in data]).items(), key=lambda x: x[1])
    res = [(x[0], len(list(x[1]))) for x in groupby(rank, key=lambda x: x[1])]
    y = [d[1] for d in res]
    # q39
    plt.hist(y)
    plt.show()

    # q39
    plt.plot(range(len(res)), y)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


q38_q39()
