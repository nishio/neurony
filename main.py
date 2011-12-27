# -*- encoding: utf-8 -*-

import numpy as np

class CharMapper(object):
    def __init__(self, chars):
        self.chars = list(chars)
        
    def __call__(self, c):
        "take a char and return an integer"
        return self.chars.index(c)


class Neurons(object):
    def __init__(self, size=100, data=None):
        self.size = size
        if data:
            self.data = data
        else:
            self.data = np.random.random(size)

    def winner_takes_all(self):
        i = (self.data + np.random.randn(self.size) / 10).argmax()
        self.clear()
        self.data[i] = 1.0
        return self

    def random(self):
        self.clear()
        self.data += np.random.random(self.size)
        return self

    def clear(self):
        self.data *= 0

    def moderate(self):
        pass

    def set(self, chars, mapper):
        self.clear()
        for c in chars:
            i = mapper(c)
            self.data[i] = 1.0

EPS = 0.0001 # small value to avoid zero-division
def normalize_prob(x):
    return ((x.T + EPS) / (x + EPS).sum(1)).T

class Network(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.weight = normalize_prob(
            np.random.random((len(src.data), len(dst.data))))

    def learn(self, alpha=0.1):
        update = np.outer(self.src.data, self.dst.data)
        MIN = update.min() - 0.001
        MAX = update.max() + 0.001
        update -= MIN
        update /= MAX - MIN
        self.weight *= 1 - alpha
        self.weight += update * alpha

    def propagate(self):
        self.dst.data += self.src.data.dot(self.weight)


DATA = u"""
りんご
青りんご
りんごジュース
みかん
オレンジ
オレンジジュース
ポンジュース
""".strip()
CHARS = set(DATA)
CHARS.remove("\n")
CHAR_MAP = CharMapper(CHARS)
DATA = DATA.split()


input = Neurons(2)
output = Neurons(2)
net = Network(input, output)
input.set("a", CharMapper("ab"))
print net.weight
output.clear()
net.propagate()
print output.data
output.winner_takes_all()
print output.data
net.learn()
print net.weight

input = Neurons(len(CHARS))
output = Neurons(2)
net = Network(input, output)
for i in range(100):
    for d in DATA:
        input.set(d, CHAR_MAP)
        output.random()
        net.learn()


for d in DATA:
    input.set(d, CHAR_MAP)
    net.propagate()
    print d, input.data
    print output.data,
    output.winner_takes_all()
    print output.data
    #net.learn()
    
