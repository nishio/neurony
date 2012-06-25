# -*- encoding: utf-8 -*-
"""
教師データを与えられるときも与えられない時もある
階層的ニューラルネットワークを作ってみる
"""
import numpy as np
from collections import Counter

# Compression

# (not implemented)

# Prediction

def enlarge(mat):
    """
    widen matrix
    >>> x = np.array([[1, 2], [3, 4]])
    >>> x
    array([[1, 2],
           [3, 4]])
    >>> enlarge(x)
    array([[ 1.,  2.,  0.],
           [ 3.,  4.,  0.],
           [ 0.,  0.,  1.]])
    """
    N, M = mat.shape
    result = np.zeros((N + 1, M + 1))
    result[0:N, 0:M] = mat
    result[N, M] = 1
    return result


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



class Prediction(object):
    def __init__(self, syms=None):
        if not syms: syms = Symbols()
        self.syms = syms
        self.mat = np.eye(syms.size)
        self._prev_input = None
        self.out_lower = None
        self.in_lower = None
        self.out_upper = None
        self.in_upper = None

    def check(self):
        """
        check if new input is in self.syms.symbols.
        If not, enlarge symbols and matrix.
        """
        s = self.in_lower
        if s not in self.syms.symbols:
            self.syms.add_a_symbol(s)
            self.mat = enlarge(self.mat)

    def step(self):
        self.check()
        prev_predict = self.out_lower
        new_input = self.in_lower
        prev_input = self._prev_input
        if (new_input != None and prev_input != None
            and prev_predict != new_input):
            # when prediction failed
            self.out_upper = (prev_input, new_input)
            # learn
            i = self.syms.sym_to_int(prev_input)
            j = self.syms.sym_to_int(new_input)
            self.mat[i, j] += 1 # just count up
        else:
            self.out_upper = None

        # predict
        arr = self.syms.sym_to_arr(new_input)
        # correct signal from upper layer
        if self.in_upper:
            (before, after) = self.in_upper
            before = self.syms.sym_to_int(before)
            after = self.syms.sym_to_int(after)
            self.mat[before, after] += 2

        p = arr.dot(self.mat)
        predict = self.syms.arr_to_sym(p)

        if self.in_upper:
            self.mat[before, after] -= 2

        self._prev_input = new_input
        self.out_lower = predict


def main(data="ABCABCBDBDBD", verbose=True):
    buf = []
    buf2 = []
    pd1 = Prediction()
    pd2 = Prediction()
    for i in range(30):
        for c in data:
            pd1.in_lower = c
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
