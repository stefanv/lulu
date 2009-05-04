import numpy as np
from numpy.testing import assert_array_equal

from lulu.ccomp import label

class TestConnectedComponents:
    def setup(self):
        self.x = np.array([[0, 0, 3, 2, 1, 9],
                           [0, 1, 1, 9, 2, 9],
                           [0, 0, 1, 9, 9, 9],
                           [3, 1, 1, 5, 3, 0]])

        self.labels = np.array([[0, 0, 1, 2, 3, 4],
                                [0, 5, 5, 4, 6, 4],
                                [0, 0, 5, 4, 4, 4],
                                [7, 5, 5, 8, 9, 10]])

    def test_basic(self):
        assert_array_equal(label(self.x), self.labels)

        # Make sure data wasn't modified
        assert self.x[0, 2] == 3

    def test_random(self):
        x = (np.random.random((20, 30)) * 5).astype(np.int)

        labels = label(x)
        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])
