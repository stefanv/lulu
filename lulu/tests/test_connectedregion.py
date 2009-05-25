from numpy.testing import *
import numpy as np

from lulu.connected_region import ConnectedRegion
import lulu.connected_region_handler as crh

class TestConnectedRegion:
    c = ConnectedRegion(shape=(5,5),
                        value=1, start_row=1,
                        rowptr=[0,4,6,8],
                        colptr=[2,3,4,5,0,3,2,5])

    dense = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0]])

    def test_basic(self):
        assert_array_equal(crh.todense(self.c), self.dense)
        assert_array_equal(crh.todense(crh.copy(self.c)), self.dense)

    def test_copy(self):
        x = [0, 1]
        c = ConnectedRegion(shape=(1,1), value=1, rowptr=[0, 2], colptr=x)
        d = crh.copy(c)
        x[1] = 0

        assert crh.todense(d) != crh.todense(c)

    def test_reshape(self):
        d = crh.copy(self.c)
        crh.reshape(d, (4, 5))
        assert_array_equal(crh.todense(d), self.dense[:4, :])

        crh.reshape(d, (5, 5))
        crh.reshape(d)
        assert_array_equal(crh.get_shape(d), (4, 5))

    def test_nnz(self):
        assert_equal(crh.nnz(self.c), 8)

    def test_start_row(self):
        c = ConnectedRegion(shape=(2,2),
                            value=1, start_row=0,
                            rowptr=[0,2],
                            colptr=[0,1])
        assert_array_equal(crh.todense(c), [[1, 0],
                                            [0, 0]])
        crh.set_start_row(c, 0)
        crh.set_start_row(c, 1)
        assert_raises(ValueError, crh.set_start_row, c, 2)

    def test_contains(self):
        d = crh.todense(self.c)
        for y, x in np.ndindex(crh.get_shape(self.c)):
            crh.contains(self.c, y, x) == d[y, x]

    def test_outside_boundary(self):
        y, x = crh.outside_boundary(self.c)
        assert_array_equal(x, [2, 4, 0, 1, 3, 5, -1, 3, 4, 0, 1, 5, 2, 3, 4])
        assert_array_equal(y, [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        c = ConnectedRegion(shape=(1, 5),
                            rowptr=[0, 2],
                            colptr=[2, 3])
        y, x = crh.outside_boundary(c)
        assert_array_equal(x, [2, 1, 3, 2])
        assert_array_equal(y, [-1, 0, 0, 1])

    def test_outside_boundary_beyond_border(self):
        c = ConnectedRegion(shape=(2, 2),
                            value=1,
                            rowptr=[0, 2, 4],
                            colptr=[0, 1, 1, 2])
        assert_array_equal(crh.todense(c), np.eye(2))

        y, x = crh.outside_boundary(c)
        assert_array_equal(y, [-1, 0, 0, 1, 1, 2])
        assert_array_equal(x, [0, -1, 1, 0, 2, 1])

    def test_value(self):
        c = ConnectedRegion(shape=(2, 2))
        assert_equal(crh.get_value(c), 0)
        crh.set_value(c, 5)
        assert_equal(crh.get_value(c), 5)
        crh.set_value(c, 0)
        assert_equal(crh.get_value(c), 0)

    def test_internal_build_ops(self):
        c = ConnectedRegion(shape=(2, 2), rowptr=[0])
        assert_equal(crh._current_row(c), 0)

        crh._append_colptr(c, 0, 1)

        crh._new_row(c)
        assert_equal(crh._current_row(c), 1)

        crh._append_colptr(c, 1, 2)
        crh._new_row(c)

        crh.set_value(c, 5)

        assert_array_equal(crh.todense(c), [[5, 0], [0, 5]])
