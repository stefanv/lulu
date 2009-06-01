import sys
sys.path.insert(0, '..')

import numpy as np
import PIL.Image as Image

import os
import time

import lulu
import lulu.connected_region_handler as crh

N = 500
runs = 1

img = Image.open(os.path.join(os.path.dirname(__file__),
                              'data/chelsea.jpg'))
img = np.array(img.convert('F')).astype(int)

# Random test data
#img = np.random.randint(255, size=(N, N))
#img[10:250, 50:100] = 10

print "Decomposing a %s matrix." % str(img.shape)

times = []
for i in range(runs):
    print "Run %s/%s" % (i + 1, runs)
    tic = time.time()
    regions = lulu.decompose(img.copy())
    toc = time.time()

    times.append(toc - tic)

print "Minimum execution time over %d runs: %ss" % (runs, min(times))

import matplotlib.pyplot as plt

out = np.zeros_like(img)

# Reconstruct
print "Reconstructing image...",
pulses = 0
for area in regions:
    pulses += len(regions[area])
    for cr in regions[area]:
        crh.set_array(out, cr, crh.get_value(cr), 1)
print "done."

plt.subplot(1, 2, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Reconstruction (%d pulses)' % pulses)

plt.suptitle('2D LULU Reconstruction')
plt.show()
