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

pd2 = brain.Prediction.init_multi_leyer(2)
pd3 = brain.Prediction.init_multi_leyer(3)
pd4 = brain.Prediction.init_multi_leyer(4)
pds = [pd2, pd3, pd4]
ng = ngram.NGram(3)

def iteration(data):
    print "%d characters" % len(data)
    starttime = clock()
    for pd in pds:
        pd.count = Counter()

    ng_count = Counter()
    for c in data:
        for pd in pds:
            pd.count.update([pd.out_lower == c])

        ng_count.update([ng.expect() == c])
        for pd in pds:
            pd.in_lower = c
            pd.step()
        ng.feed(c)

    N = float(len(data))
    for pd in pds:
        print "%s: correct %.1f%% mem %d" % (
            pd, (100 * pd.count[True] / N), pd.size())

    print "NGram(3): correct %.1f%% mem %d" % ((100 * ng_count[True] / N), ng.size())
    print clock() - starttime, "sec"
    print


timemachine = open("timemachine.txt")
for i in range(5):
    data = timemachine.read(100)
    iteration(data)
