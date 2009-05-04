import numpy as np
import matplotlib.pyplot as plt

import lulu.lulu_base as base

c = base.ConnectedRegion(shape=(5,5),
                         value=1, start_row=1,
                         rowptr=[0,4,6,10,14],
                         colptr=[2,3,4,5,0,5,0,1,2,5,0,2,3,5])

print c.todense()

dense = np.zeros((7,7,3))
dense[1:6, 1:6, 0] = c.todense()

plt.subplot(1, 3, 1)
plt.imshow(dense, interpolation='nearest')
plt.title('Connected region')

ii, jj = c.outside_boundary()
dense_outside = dense.copy()
for i, j in zip(ii, jj):
    dense_outside[i + 1, j + 1] = [0, 1, 0]

plt.subplot(1, 3, 2)
plt.imshow(dense_outside, interpolation='nearest')
plt.title('Outside boundary')

ii, jj = c.inside_boundary()
dense_inside = dense.copy()
for i, j in zip(ii, jj):
    dense_inside[i + 1, j + 1] = [0, 0, 1]

plt.subplot(1, 3, 3)
plt.imshow(dense_inside, interpolation='nearest')
plt.title('Inside boundary')

plt.show()
