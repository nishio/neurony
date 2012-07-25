# -*- encoding: utf-8 -*-
"""
教師データを与えられるときも与えられない時もある
階層的ニューラルネットワークを作ってみる
"""
import numpy as np
from collections import Counter, defaultdict

# Compression

# (not implemented)

# Prediction

# Mapper
class Symbols(object):
    def __init__(self, data=[]):
        c = Counter(data)
        # self.symbols should not be a set, because index is important
        self.symbols = [x[0] for x in c.most_common()]
        self.size = len(self.symbols)

    def sym_to_int(self, sym):
        return self.symbols.index(sym)

    def int_to_sym(self, i):
        return self.symbols[i]

    def sym_to_arr(self, sym):
        return np.array([int(x == sym) for x in self.symbols])

    def arr_to_sym(self, arr):
        return self.symbols[arr.argmax()]

    def add_a_symbol(self, sym):
        assert sym not in self.symbols
        self.symbols.append(sym)
        self.size += 1

class Eye(defaultdict):
    def __missing__(self, key):
        self[key] = Counter([key])
        return self[key]

class Prediction(object):
    """
    a model of Cerebral cortex
    """
    def __init__(self, syms=None):
        if not syms: syms = Symbols()
        self.syms = syms
        self.data = Eye()
        self._prev_input = None
        self.out_lower = None
        self.in_lower = None
        self.out_upper = None
        self.in_upper = None
        self.upper_layer = None
        self.num_layer = 1

    @staticmethod
    def init_multi_leyer(num):
        top = ret = Prediction()
        ret.num_layer = num
        num -= 1
        while num > 0:
            new_top = Prediction()
            top.set_upper_layer(new_top)
            top = new_top
            num -= 1

        return ret

    def __str__(self):
        return "Brain(%s)" % self.num_layer

    def set_upper_layer(self, upper):
        self.upper_layer = upper

    def step(self):
        # upper layer
        if self.upper_layer:
            self.in_upper = self.upper_layer.out_lower

        prev_predict = self.out_lower
        new_input = self.in_lower
        prev_input = self._prev_input
        if (new_input != None and prev_input != None
            and prev_predict != new_input):
            # when prediction failed
            self.out_upper = (prev_input, new_input)
            # learn
            self.data[prev_input].update([new_input])
        else:
            self.out_upper = None

        # predict
        # correct signal from upper layer
        to_decrement = False
        if self.in_upper:
            (before, after) = self.in_upper
            self.data[before].update({after: 2})

        predict, _count = self.data[new_input].most_common(1)[0]

        if self.in_upper:
            self.data[before].update({after: -2})


        self._prev_input = new_input
        self.out_lower = predict

        # propagate to upper layer
        if self.out_upper and self.upper_layer:
            self.upper_layer.in_lower = self.out_upper
            self.upper_layer.step()

    def size(self):
        ret = sum(len(self.data[k]) for k in self.data)
        if self.upper_layer:
            ret += self.upper_layer.size()
        return ret


def main(data="ABCABCBDBDBD", verbose=True):
    buf = []
    buf2 = []
    pd1 = Prediction()
    pd2 = Prediction()
    for i in range(30):
        for c in data:
            pd1.in_lower = c
            pd1.in_upper = pd2.out_lower
            pd1.step()
            if pd1.out_upper:
                pd2.in_lower = pd1.out_upper
                pd2.step()
            if verbose:
                print "%s\t%s\t%s\t%s\t%s" % (
                    c, pd1.out_lower, pd1.out_upper,
                    pd2.out_lower, pd2.out_upper)
            buf.append(pd1.out_upper)
            buf2.append(pd2.out_upper)
    return buf, buf2


def _test():
    import doctest
    doctest.testmod();


if globals().has_key("In"):
    # when execfile-ed from ipython env
    main()
    #main(file("brain.py").read(), False)
    pass
elif __name__ == "__main__":
    # when executed from console, run test
    _test()
