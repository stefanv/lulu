import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

import cython

cdef class ConnectedRegion:
    """
    A connected region is stored in a modified Compressed
    Sparse Row matrix format.

    Since the region is connected, we only have to store one value.

    `start_row` is the first row in which values are found.  In each
    row ``start_row + i``, the columns ``column_start[i]`` through
    ``column_end[i] - 1`` are filled with value.

    """
    cdef int value
    cdef int start_row
    cdef list column_start, column_end
    cdef int nnz # nr of non-zeroes

    def __init__(self, value=0, start_row=0, column_start=[], column_end=[]):
        self.value = value
        self.start_row = start_row
        self.column_start = column_start
        self.column_end = column_end
        self.nnz = len(column_start)

    @cython.boundscheck(False)
    def todense(self, shape):
        """Convert the connected region to a dense matrix of the
        given shape.

        """
        cdef np.ndarray[np.int_t, ndim=2] out = np.zeros(shape, dtype=np.int)

        cdef int i, j, start, end

        for i in range(self.nnz):
            start = self.column_start[i]
            end = self.column_end[i]
            for j in range(start, end):
                out[i + self.start_row, j] = self.value

        return out

