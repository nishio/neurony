# -*- coding: utf-8 -*-
from collections import defaultdict, Counter, deque
class NGram(object):
    """
    >>> ngram = NGram(3)
    >>> ngram.feed_all("ABABABABABABAB")
    >>> ngram.expect()
    'A'
    >>> ngram.feed('A')
    >>> ngram.expect()
    'B'
    """
    def __init__(self, n):
        self.n = n
        self.data = defaultdict(Counter)
        self.backlog = deque(["BOS"] * n)
        self.freq_counter = Counter()

    def feed(self, c):
        self.data[tuple(self.backlog)].update([c])
        self.backlog.append(c)
        self.backlog.popleft()
        self.freq_counter.update([c])

    def feed_all(self, string):
        for c in string:
            self.feed(c)

    def expect(self):
        mc = self.data[tuple(self.backlog)].most_common(1)
        if len(mc) > 0:
            return mc[0][0]
        # TODO smoothing
        return None

    def size(self):
        return sum(len(self.data[k]) for k in self.data)

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
