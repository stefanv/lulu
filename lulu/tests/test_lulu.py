from numpy.testing import *
import numpy as np

import lulu

class TestLULU:
    img = np.zeros((5, 5)).astype(int)
    img[0, 0:5] = 0
    img[:, 4] = 1
    img[1:3, 1:4] = 2
    """
    [[0 0 0 0 1]
     [0 2 2 2 1]
     [0 2 2 2 1]
     [0 0 0 0 1]
     [0 0 0 0 1]]
    """

    def test_connected_regions(self):
        regions = lulu.connected_regions(self.img)

        assert_equal(len(regions), 3)

        regions[0].value = 5
        assert_array_equal(regions[0].todense(),
                           [[5, 5, 5, 5, 0],
                            [5, 0, 0, 0, 0],
                            [5, 0, 0, 0, 0],
                            [5, 5, 5, 5, 0],
                            [5, 5, 5, 5, 0]])

        assert_array_equal(regions[1].todense(),
                           [[0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1]])

        print regions[2].todense()
        assert_array_equal(regions[2].todense(),
                           [[0, 0, 0, 0, 0],
                            [0, 2, 2, 2, 0],
                            [0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
