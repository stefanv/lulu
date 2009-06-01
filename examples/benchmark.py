import sys
sys.path.insert(0, '..')

import numpy as np
import PIL.Image as Image
import os
import time

import lulu

N = 500
runs = 1

img = Image.open(os.path.dirname(__file__) + './data/chelsea.jpg')
img = np.array(img.convert('F')).astype(int)

#img = np.random.randint(255, size=(N, N))
#img[10:250, 50:100] = 10

print "Decomposing a %s matrix." % str(img.shape)

times = []
for i in range(runs):
    print "Run %s/%s" % (i, runs)
    tic = time.time()
    out = lulu.decompose(img)
    toc = time.time()

    times.append(toc - tic)

print "Minimum execution time over three runs: %ss" % min(times)
print "Maximum execution time over three runs: %ss" % max(times)

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
plt.subplot(2, 1, 2)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.show()
