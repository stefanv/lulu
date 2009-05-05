import sys
sys.path.insert(0, '..')

import numpy as np
import time

import lulu

N = 500
runs = 3

print "Decomposing a (%s, %s) matrix." % (N, N)

x = np.random.randint(255, size=(N, N))
times = []
for i in range(runs):
    print "Run %s/%s" % (i, runs)
    tic = time.time()
    out = lulu.connected_regions(x)
    toc = time.time()

    times.append(toc - tic)

print "Minimum execution time over three runs: %ss" % min(times)
print "Connected regions found:", len(out)
