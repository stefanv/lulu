from numpy.testing import *

import lulu.lulu_base as base

def test_create():
    c = base.ConnectedRegion()

def test_todense():
    c = base.ConnectedRegion(value=1, start_row=1,
                             column_start=[2, 0, 3],
                             column_end=[3, 3, 5])
    assert_array_equal(c.todense((5, 5)),
                       [[0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0]])

