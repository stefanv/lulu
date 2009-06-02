import sys
sys.path.insert(0, '..')

import numpy as np
import PIL.Image as Image

import os
import time

import lulu
import lulu.connected_region_handler as crh

N = 300
runs = 1

if len(sys.argv) == 2:
    img = Image.open(sys.argv[1])
else:
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

def memory_use(regions):
    """Estimate the memory use of the given regions.

    """
    mem = 0
    for area in regions:
        for r in regions[area]:
            mem += crh.mem_use(r)

    return mem


print "-"*78
print "Reconstructing image...",
out, areas, area_count, pulses = lulu.reconstruct(regions, img.shape)
print "done."
print "Reconstructed from %d pulses." % pulses
print "Estimated memory use: %d bytes" % memory_use(regions)
print "-"*78

plt.subplot(2, 2, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Reconstruction (%d pulses)' % pulses)

plt.subplot(2, 2, 4)
s = np.cumsum(area_count)
midpt = (s[-1] + s[0])/2.
ind = np.argmin(np.abs(s - midpt))
plt.plot([areas[ind]], [area_count[ind]], 'r.')

areas = areas[:ind*3]
area_count = area_count[:ind*3]

plt.plot(areas, area_count)
plt.xlabel('Pulse Area')
plt.ylabel('Number of Pulses')
plt.title('Histogram of Pulse Areas (up to area %d)' % (ind*3))

print "-"*78
print "Thresholded reconstruction...",
out, areas, area_count, pulses = \
     lulu.reconstruct(regions, img.shape, min_area=areas[ind])
print "done."
print "Reconstructed from %d pulses." % pulses

for area in regions:
    if area < areas[ind]:
        regions[area] = []

print "Estimated memory use: %d bytes" % memory_use(regions)
print "-"*78


plt.subplot(2, 2, 3)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Reconstruction with areas >= %d (%d pulses)' % (areas[ind], pulses))

plt.suptitle('2D LULU Reconstruction')
plt.show()
