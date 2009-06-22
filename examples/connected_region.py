import sys
sys.path.extend('..')

import numpy as np
import matplotlib.pyplot as plt

from lulu import ConnectedRegion
from lulu import connected_region_handler as crh

c = ConnectedRegion(shape=(5,5),
                    value=1, start_row=1,
                    rowptr=[0,4,6,10,14],
                    colptr=[2,3,4,5,0,5,0,1,2,5,0,2,3,5])

print crh.todense(c)

dense = np.zeros((7,7,3))
dense[1:6, 1:6, 0] = crh.todense(c)

plt.subplot(1, 2, 1)
plt.imshow(dense, interpolation='nearest')
plt.title('Connected region')
plt.xticks([])
plt.yticks([])

ii, jj = crh.outside_boundary(c)
dense_outside = dense.copy()
for i, j in zip(ii, jj):
    dense_outside[i + 1, j + 1] = [0, 1, 0]

plt.subplot(1, 2, 2)
plt.imshow(dense_outside, interpolation='nearest')
plt.title('Outside boundary')
plt.xticks([])
plt.yticks([])

plt.show()
