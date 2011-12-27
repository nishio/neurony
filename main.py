# -*- encoding: utf-8 -*-

import numpy as np
from random import shuffle

class CharMapper(object):
    def __init__(self, chars):
        self.chars = list(chars)
        self.size = len(self.chars)
        
    def to_int(self, c):
        "take a char and return an integer"
        return self.chars.index(c)

    def to_obj(self, i):
        return self.chars[i]


class Neurons(object):
    def __init__(self, size=100, data=None, mapper=None):
        if mapper:
            self.mapper = mapper
            self.size = mapper.size
        else:
            self.size = size

        if data:
            self.data = data
        else:
            #self.data = np.random.random(self.size)
            self.data = np.repeat(0.5, self.size)

    def find_winner(self):
        return (self.data + np.random.randn(self.size) / 1000).argmax()

    def winner_takes_all(self):
        i = self.find_winner()
        self.clear()
        self.data[i] = 1.0
        return self

    def random(self):
        self.clear()
        self.data += np.random.random(self.size)
        return self

    def clear(self):
        self.data *= 0

    def activate(self):
        self.clear()
        self.data += 1

    def set(self, chars, mapper=None):
        if not mapper: mapper = self.mapper
        self.clear()
        for c in chars:
            i = mapper.to_int(c)
            self.data[i] = 1.0

    def get(self, mapper=None):
        if not mapper: mapper = self.mapper
        i = self.find_winner()
        return mapper.to_obj(i)


EPS = 0.0001 # small value to avoid zero-division
def normalize_prob(x):
    return ((x.T + EPS) / (x + EPS).sum(1)).T

class Network(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.weight = normalize_prob(
            #np.random.random((len(src.data), len(dst.data))))
            np.ones((src.size, dst.size)))

    def learn(self, alpha=0.1):
        update = np.outer(self.src.data, self.dst.data)
        self.weight[self.src.data > 0.5] *= 1 - alpha
#        self.weight[self.src.data > 0.5] += update[self.src.data > 0.5] * alpha
        self.weight[self.src.data > 0.5] += self.dst.data * alpha

        self.weight[self.src.data < 0.5] *= 1 - alpha
        self.weight[self.src.data < 0.5] += (1 - self.dst.data) * alpha

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

DATA = u"""
うさぎ
ねずみ
かささぎ
ねこ
カッパ
パンダ
ダルメシアン
""".strip()


CHARS = set(DATA)
CHARS.remove("\n")
CHAR_MAP = CharMapper(CHARS)
DATA = DATA.split()

input = Neurons(mapper=CharMapper(CHARS))
output = Neurons(mapper=CharMapper("ox"))
net = Network(input, output)

def perceptron():
    answer = "oooxxxx"
    for i in range(100):
        for d, ans in zip(DATA, answer):
            input.set(d)
            output.set(ans)
            net.learn()

    for d, ans in zip(DATA, answer):
        input.set(d)
        net.propagate()
        print d, ans, output.get()


def kmeans():
    for d in DATA:
        input.set(d)
        output.activate()
        net.learn()

    for i in range(100):
        for d in DATA:
            input.set(d)
            net.propagate()
            output.winner_takes_all()
            net.learn()

    for d in DATA:
        input.set(d)
        net.propagate() 
        print d, output.get()
