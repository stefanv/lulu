import sys
sys.path.insert(0, '..')

from enthought.mayavi import mlab
import numpy as np

from demo import load_image

import lulu
import lulu.connected_region_handler as crh

img = load_image('chelsea_small.jpg')

print "Decomposing a %s image." % str(img.shape)
regions = lulu.decompose(img)

value_maxes = []
height = 0
for area in sorted(regions.keys()):
    pulses = regions[area]

    if len(pulses) == 0 or area < 280 or area > 300:
        continue

    values = [crh.get_value(p) for p in pulses]
    height_diff = max(values) - min(values)
    value_maxes.append(height_diff)
    centre = height + height_diff / 2.0

    pulse_values = np.zeros_like(img)
    for p in pulses:
        crh.set_array(pulse_values, p, crh.get_value(p))

    y, x = np.where(pulse_values)
    s = pulse_values[y, x]

    mlab.barchart(x, y, [height + centre] * len(s), s,
                  opacity=1.0, scale_factor=1.5)

    height += height_diff + 0.5

scene = mlab.get_engine().scenes[0]
scene.scene.parallel_projection = True
mlab.show()
