from demo import load_image

import matplotlib.pyplot as plt

import numpy as np

import lulu
import lulu.connected_region_handler as crh

img = load_image('truck_and_apcs_small.jpg')

pulses = lulu.decompose(img)

areas = sorted(pulses.keys())
cumulative_volume = []
volumes = []
reconstruction = np.zeros_like(img)
for area in areas:
    area_volume = 0
    for cr in pulses[area]:
        area_volume += crh.nnz(cr) * abs(crh.get_value(cr))
        crh.set_array(reconstruction, cr, abs(crh.get_value(cr)), 'add')
    cumulative_volume.append(np.sum(reconstruction))
    volumes.append(area_volume)

total_volume = np.sum(reconstruction)
cumulative_volume = np.array(cumulative_volume)
cumulative_volume = 1 - cumulative_volume/float(total_volume)

plt.subplot(1, 3, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

plt.subplot(1, 3, 2)
plt.plot(areas[:-10], volumes[:-10], 'x-')
#plt.xlim(plt.xlim()[::-1])
plt.title('Level Volumes')
plt.xlabel('Pulse area')
plt.ylabel('Pulse volume')

plt.subplot(1, 3, 3)
plt.plot(areas[:-10], cumulative_volume[:-10], 'x-')

plt.show()
