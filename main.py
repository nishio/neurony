# -*- encoding: utf-8 -*-

import numpy as np

class CharMapper(object):
    def __init__(self, chars):
        self.chars = list(chars)
        self.size = len(self.chars)
        
    def to_int(self, c):
        "take a char and return an integer"
        return self.chars.index(c)

    def to_obj(self, i):
        return self.chars[i]

    def set(self, target, chars):
        target.clear()
        for c in chars:
            i = self.to_int(c)
            target.data[i] = 1.0

    def get(self, target):
        i = target.find_winner()
        return self.to_obj(i)
            

class LEDMapper(object):
    size = 25
    def set(self, target, pattern):
        target.clear()
        pattern = pattern.replace("\n", "")
        for i in range(len(pattern)):
            if pattern[i] == "+":
                target.data[i] = 1.0

    def get(self, target):
        buf = []
        for i in range(5):
            for j in range(5):
                if target.data[i * 5 + j] > 0.5:
                    c = "+"
                else:
                    c = "."
                buf.append(c)
            buf.append("\n")
        return "".join(buf)

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

    def set(self, *args, **kw):
        self.mapper.set(self, *args, **kw)

    def get(self, *args, **kw):
        return self.mapper.get(self, *args, **kw)


EPS = 0.0001 # small value to avoid zero-division
def normalize_prob(x):
    return ((x.T + EPS) / (x + EPS).sum(1)).T

class Network(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.weight = normalize_prob(
            np.ones((src.size, dst.size)))

    def learn(self, alpha=0.1, both=False):
        update = np.outer(self.src.data, self.dst.data)
        self.weight[self.src.data > 0.5] *= 1 - alpha
        self.weight[self.src.data > 0.5] += self.dst.data * alpha
        if both:
            self.weight[self.src.data < 0.5] *= 1 - alpha
            self.weight[self.src.data < 0.5] += (1 - self.dst.data) * alpha

    def propagate(self):
        self.dst.data += self.src.data.dot(self.weight)


DATA = """
+++++
+...+
+...+
+...+
+++++

..+..
..+..
..+..
..+..
..+..

++++.
+..+.
+..+.
+..+.
++++.

..+..
.++..
..+..
..+..
..+..

.++++
.+..+
.+..+
.+..+
.++++

..+..
.++..
..+..
..+..
.+++.
""".strip().split("\n\n")

input = Neurons(mapper=LEDMapper())
output = Neurons(mapper=CharMapper("01"))
net = Network(input, output)

def perceptron():
    answer = "010101"
    for i in range(100):
        for d, ans in zip(DATA, answer):
            input.set(d)
            output.set(ans)
            net.learn()

    for d, ans in zip(DATA, answer):
        output.clear()
        input.set(d)
        net.propagate()
        print d, ans, output.get()
        print


def kmeans():
    for i in range(100):
        for d in DATA:
            input.set(d)
            net.propagate()
            output.winner_takes_all()
            net.learn(both=True)

    for d in DATA:
        output.clear()
        input.set(d)
        net.propagate() 
        print d, output.get()
        print

