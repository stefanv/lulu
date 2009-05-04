from numpy.testing import *
import numpy as np

import lulu.lulu_base as base

class TestConnectedRegion:
    c = base.ConnectedRegion(shape=(5,5),
                             value=1, start_row=1,
                             rowptr=[0,4,6,8],
                             colptr=[2,3,4,5,0,3,2,5])

    dense = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0]])

    def test_basic(self):
        assert_array_equal(self.c.todense(), self.dense)
        assert_array_equal(self.c.copy().todense(), self.dense)

    def test_reshape(self):
        d = self.c.copy()
        d.reshape((4,5))
        assert_array_equal(d.todense(), self.dense[:4, :])

    def test_nnz(self):
        assert_equal(self.c.nnz, 8)

    def test_start_row(self):
        c = base.ConnectedRegion(shape=(2,2),
                                 value=1, start_row=0,
                                 rowptr=[0,2],
                                 colptr=[0,1])
        assert_array_equal(c.todense(), [[1, 0],
                                         [0, 0]])
        c.start_row = 0
        c.start_row = 1
        assert_raises(ValueError, c.set_start_row, 2)

    def test_contains(self):
        d = self.c.todense()
        for y, x in np.ndindex(self.c.shape):
            self.c.contains(y, x) == d[y, x]

    def test_outside_boundary(self):
        y, x = self.c.outside_boundary()
        assert_array_equal(x, [2, 4, 0, 1, 3, 5, -1, 3, 4, 0, 1, 5, 2, 3, 4])
        assert_array_equal(y, [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

    def test_outside_boundary_beyond_border(self):
        c = base.ConnectedRegion(shape=(2, 2),
                                 value=1,
                                 rowptr=[0, 2, 4],
                                 colptr=[0, 1, 1, 2])
        assert_array_equal(c.todense(), np.eye(2))

        y, x = c.outside_boundary()
        assert_array_equal(y, [-1, 0, 0, 1, 1, 2])
        assert_array_equal(x, [0, -1, 1, 0, 2, 1])
