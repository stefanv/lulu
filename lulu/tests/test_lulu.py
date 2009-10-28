from numpy.testing import *
import numpy as np

import lulu
import lulu.connected_region_handler as crh

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
        labels, regions = lulu.connected_regions(self.img)

        assert_array_equal(labels, self.img)

        assert_equal(len(regions), 3)

        crh.set_value(regions[0], 5)
        assert_array_equal(crh.todense(regions[0]),
                           [[5, 5, 5, 5, 0],
                            [5, 0, 0, 0, 0],
                            [5, 0, 0, 0, 0],
                            [5, 5, 5, 5, 0],
                            [5, 5, 5, 5, 0]])

        assert_array_equal(crh.todense(regions[1]),
                           [[0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1]])

        assert_array_equal(crh.todense(regions[2]),
                           [[0, 0, 0, 0, 0],
                            [0, 2, 2, 2, 0],
                            [0, 2, 2, 2, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

class TestReconstruction:
    def test_basic(self):
        img = np.random.randint(255, size=(10, 15))

        pulses = lulu.decompose(img)
        img_, areas, area_count = lulu.reconstruct(pulses, img.shape)
        assert_array_equal(img_, img)

