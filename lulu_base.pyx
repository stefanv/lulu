import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

import cython
import copy

cdef class ConnectedRegion:
    """
    A connected region is stored in a modified Compressed
    Sparse Row matrix format.

    Since the region is connected, we only have to store one value.
    Along a single row, connected regions are stored as index pairs, e.g.

    ---00-000-- would be represented as [3, 5, 6, 9]

    Attributes
    ----------
    rowptr : list of int
        `rowptr[i]` tells us where in `colptr` the elements of row i are
        described
    colptr : list of int
        Always contains 2N elements, where N are the number of connected
        regions (see description above).

    """
    cdef int value
    cdef int start_row
    cdef list rowptr, colptr
    cdef int _nnz
    cdef tuple _shape

    def __init__(self, shape=None, value=0, start_row=0, rowptr=[], colptr=[]):
        self._shape = shape
        self.value = value
        self.start_row = start_row
        self.rowptr = rowptr
        self.colptr = colptr

    @cython.boundscheck(True)
    def todense(self):
        """Convert the connected region to a dense array.

        """
        shape = self._shape
        if shape is None:
            shape = self._minimum_shape()

        cdef np.ndarray[np.int_t, ndim=2] out = np.zeros(shape, dtype=np.int)

        cdef int i, j, k, start, end

        for i in range(len(self.rowptr) - 1):
            for j in range((self.rowptr[i + 1] - self.rowptr[i]) / 2):
                start = self.colptr[self.rowptr[i] + 2*j]
                end = self.colptr[self.rowptr[i] + 2*j + 1]
                for k in range(start, end):
                    out[i + self.start_row, k] = self.value

        return out

    @property
    def nnz(self):
        """Return the number of non-zero elements.

        """
        cdef int nnz = 0

        for i in range(len(self.colptr) / 2):
            nnz += self.colptr[2*i + 1] - self.colptr[2*i]

        return nnz

    def _minimum_shape(self):
        """Return the minimum shape into which the connected region can fit.

        """
        return (self.start_row + len(self.rowptr) - 1, max(self.colptr))

    def reshape(self, shape):
        """Set the shape of the connected region.

        Useful when converting to dense.
        """
        if (shape >= self._minimum_shape()):
            self._shape = shape
        else:
            raise ValueError("Minimum shape is %s." % self._minimum_shape())

    def copy(self):
        """Return a deep copy of the connected region.

        """
        return ConnectedRegion(shape=self._shape, value=self.value,
                               start_row=self.start_row,
                               rowptr=self.rowptr, colptr=self.colptr)
