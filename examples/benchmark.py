import sys
sys.path.insert(0, '..')

from demo import load_image

import time
import lulu

img = load_image()

print "Decomposing a %s matrix." % str(img.shape)

tic = time.time()
regions = lulu.decompose(img)
toc = time.time()

print "Execution time: %.2fs" % (toc - tic)
