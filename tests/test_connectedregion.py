from numpy.testing import *
import numpy as np

import lulu.lulu_base as base

class TestToDense:
    c = base.ConnectedRegion(shape=(5,5),
                             value=1, start_row=1,
                             rowptr=[0,4,6,8],
                             colptr=[2,3,4,5,0,3,3,5])

    dense = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0]])

    def test_basic(self):
        assert_array_equal(self.c.todense(), self.dense)
        assert_array_equal(self.c.copy().todense(), self.dense)

    def test_reshape(self):
        d = self.c.copy()
        d.reshape((4,5))
        assert_array_equal(d.todense(), self.dense[:4, :])

    def test_nnz(self):
        assert_equal(self.c.nnz, 7)

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
