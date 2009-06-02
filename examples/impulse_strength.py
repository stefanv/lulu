"""
Illustrate how the impulse strength varies across the image.

"""

from demo import load_image

import numpy as np
import matplotlib.pyplot as plt

import lulu
import lulu.connected_region_handler as crh

img = load_image()

pulses = lulu.decompose(img.copy())

out = np.zeros(img.shape, dtype=int)
for area in pulses:
    for cr in pulses[area]:
        crh.set_array(out, cr, abs(crh.get_value(cr)), 'add')

plt.subplot(1, 2, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(out, interpolation='nearest', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.title('Impulse Strength')
plt.show()
