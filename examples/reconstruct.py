import sys
sys.path.insert(0, '..')

from demo import load_image

import numpy as np
import matplotlib.pyplot as plt

import os
import time

import lulu
import lulu.connected_region_handler as crh

img = load_image()

print("Decomposing a %s matrix." % str(img.shape))

tic = time.time()
regions = lulu.decompose(img.copy())
toc = time.time()

print("Execution time: %.2fs" % (toc - tic))

print("-"*78)
print("Reconstructing image...", end=None)
out, areas, area_count = lulu.reconstruct(regions, img.shape)
print("done.")
print("Reconstructed from %d pulses." % sum(area_count))
print("-"*78)

plt.subplot(2, 2, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Reconstruction (%d pulses)' % sum(area_count))

plt.subplot(2, 2, 4)
s = np.cumsum(area_count)
midpt = (s[-1] + s[0])/2.
ind = np.argmin(np.abs(s - midpt))
plt.plot([areas[ind]], [area_count[ind]], 'r.', markersize=10)

areas = areas[:ind*3]
area_count = area_count[:ind*3]

#plt.fill_between(areas[ind:], area_count[ind:], alpha=0.3)

plt.plot(areas, area_count)
plt.xlabel('Pulse Area')
plt.ylabel('Number of Pulses')
plt.title('Histogram of Pulse Areas (up to area %d)' % (ind*3))

print("-"*78)
print("Thresholded reconstruction...", end=None)
out, areas, area_count = \
     lulu.reconstruct(regions, img.shape, min_area=areas[ind])
print("done.")
print("Reconstructed from %d pulses." % sum(area_count))

for area in regions:
    if area < areas[ind]:
        regions[area] = []

plt.subplot(2, 2, 3)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Reconstruction with areas >= %d (%d pulses)' % \
          (areas[ind], sum(area_count)))

plt.suptitle('2D LULU Reconstruction')
plt.show()
