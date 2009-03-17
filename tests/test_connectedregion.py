from numpy.testing import *
import numpy as np

import lulu.lulu_base as base

def test_create():
    c = base.ConnectedRegion()

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
