# -*- coding: utf-8 -*-
"""
実験2
タイムマシンデータを1万文字ずつ順に食わせる
その過程でのメモリ消費量と正解率の変化を観察する
"""
import ngram
import brain
from collections import Counter
from time import clock

timemachine = open("timemachine.txt").read()[:300]

def main(data):
    print "%d characters" % len(data)
    starttime = clock()
    pd1 = brain.Prediction()
    pd2 = brain.Prediction()
    pd1.set_upper_layer(pd2)
    pd_count = Counter()
    ng = ngram.NGram(3)
    ng_count = Counter()
    for c in data:
        pd_count.update([pd1.out_lower == c])
        ng_count.update([ng.expect() == c])
        pd1.in_lower = c
        pd1.step()
        ng.feed(c)

    N = float(len(data))
    print "Brain(2): %.1f%%" % (100 * pd_count[True] / N), pd_count[True]
    print "NGram(3): %.1f%%" % (100 * ng_count[True] / N), ng_count[True], ng.size()
    print clock() - starttime, "sec"
    print

main(timemachine)
