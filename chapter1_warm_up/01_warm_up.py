from re import sub


def q0():
    print("stressed"[::-1])


def q1():
    print("パタトクカシーー"[0::2])


def q2():
    p = "パトカー"
    t = "タクシー"
    print(''.join([a+b for a, b in zip(p, t)]))


def q3():
    sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    ret = [(w[0] if i in [1, 5, 6, 7, 8, 9, 15, 16, 19] else w[0:2], i) for i, w in enumerate(sentence.split(" "), start=1)]
    print(dict(ret))


def q4_1():
    sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    print([len(w) if w.isalnum() else len(w) - 1 for w in sentence.split(" ")])


def q4_2():
    sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    sen2 = sub("[\.,]", "", sentence)
    print([len(w) for w in sen2.split(" ")])


def ngram(n, l):
    return [i for i in zip(*[l[i:] for i in range(n)])]


def q5():
    sentence = "I am an NLPer"

    print(ngram(2, sentence))
    print(ngram(2, sentence.split(" ")))
    print(ngram(3, sentence))
    print(ngram(3, sentence.split(" ")))


def q6():
    st1 = "paraparaparadise"
    st2 = "paragraph"
    set1 = set(ngram(2, st1))
    set2 = set(ngram(2, st2))

    print(set1 | set2)
    print(set1 & set2)
    print(set1 - set2)
    print(('s', 'e') in set1)
    print(('s', 'e') in set2)


def q7():
    def a(x, y, z): return print(f'{x}時の{y}は{z}')
    a(12, "気温", 22.4)


def q8():
    def cipher(l):
        return ''.join([chr(219 - ord(c)) if c.islower() else c for c in l])

    mes = "Unsupported operand type(s) for -: 'int' and 'str'"
    print(cipher(mes))
    print(cipher(cipher(mes)))


def q9():
    from random import sample
    mes = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

    def func(w):
        if len(w) <= 4:
            return w
        return ''.join([w[0], *sample(w[1:len(w)-1], len(w)-2), w[-1]])

    print(' '.join([func(w) for w in mes.split(' ')]))
