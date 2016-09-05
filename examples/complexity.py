import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

import time
import lulu


sizes = []
times = []
for n in [16, 32, 64, 128, 256, 512, 1024]:
    print("DPT of size (%s, %s)" % (n, n))
    x = np.random.randint(255, size=(n, n))
    tic = time.time()
    lulu.decompose(x)
    toc = time.time()

    sizes.append(n*n)
    times.append(toc - tic)

plt.plot(sizes, times, '-x')
plt.title('Execution times for 2D DPT')
plt.xlabel('Size, i.e. NM for an NxM image')
plt.ylabel('Execution time')

plt.show()
