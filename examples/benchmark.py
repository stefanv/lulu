import sys
sys.path.insert(0, '..')

from demo import load_image

import numpy as np

import os
import time

import lulu
import lulu.connected_region_handler as crh

img = load_image()

print "Decomposing a %s matrix." % str(img.shape)

tic = time.time()
regions = lulu.decompose(img.copy())
toc = time.time()

print "Execution time: %.2fs" % (toc - tic)
