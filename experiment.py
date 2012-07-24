"""
data feeder
"""
import ngram
import brain
from collections import Counter
from time import clock

abra = "abracadabra abracadabra abracadabra abracadabra abracadabra"

lorem = """Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""


brainpy = open("brain.py").read()
timemachine = open("timemachine.txt").read()[:10000]

repeat = lorem * 5
data = ""
def main(data=data):
    print "%d characters" % len(data)
    starttime = clock()
    pd1 = brain.Prediction()
    pd2 = brain.Prediction()
    pd_count = Counter()
    ng = ngram.NGram(3)
    ng_count = Counter()
    for c in data:
        pd_count.update([pd1.out_lower == c])
        ng_count.update([ng.expect() == c])

        pd1.in_lower = c
        pd1.in_upper = pd2.out_lower
        pd1.step()
        if pd1.out_upper:
            pd2.in_lower = pd1.out_upper
            pd2.step()

        ng.feed(c)

    N = float(len(data))
    print "Brain: %.1f%%" % (100 * pd_count[True] / N), pd_count[True]
    print "NGram: %.1f%%" % (100 * ng_count[True] / N), ng_count[True]
    print clock() - starttime, "sec"
    print

print "abracadabra"
main(abra)
print "Lorem"
main(lorem)
print "brain.py"
main(brainpy)
print "Lorem * 5"
main(repeat)
print "timemachine"
main(timemachine)

