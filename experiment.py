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

pd1 = brain.Prediction.init_multi_leyer(1)
pd2 = brain.Prediction.init_multi_leyer(2)
pd3 = brain.Prediction.init_multi_leyer(3)
pd4 = brain.Prediction.init_multi_leyer(4)
pds = [pd1, pd2, pd3, pd4]
ng = ngram.NGram(2)

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
    line = []
    for pd in pds:
        correct = 100 * pd.count[True] / N
        size = pd.size()
        print "%s: correct %.1f%% mem %d" % (
            pd, correct, size)
        line.append(correct)
        line.append(size)

    correct = 100 * ng_count[True] / N
    size = ng.size()
    print "NGram(3): correct %.1f%% mem %d" % (correct, size)
    line.append(correct)
    line.append(size)
    print clock() - starttime, "sec"
    print
    buf.append(line)

timemachine = open("timemachine.txt")
buf = []
for i in range(20):
    data = timemachine.read(10000)
    iteration(data)

print "\n".join(",".join(map(str, line)) for line in buf)
