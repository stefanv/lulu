import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

import cython
import copy

cdef class ConnectedRegion:
    """
    A 4-connected region is stored in a modified Compressed
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
    cdef int _start_row
    cdef list rowptr, colptr
    cdef int _nnz
    cdef tuple _shape

    def __init__(self, shape, value=0, start_row=0, rowptr=[], colptr=[]):
        if shape is None:
            raise ValueError("Shape must be specified.")

        self._shape = shape
        self.value = value
        self.rowptr = rowptr
        self.colptr = colptr
        self.start_row = start_row

    def _iterate_rows(self):
        """For each row, return the connected columns as

        (r, start, end)
        """
        cdef list out = []
        cdef int r, rp, c

        for r, rp in enumerate(range(len(self.rowptr) - 1)):
            for c in range((self.rowptr[rp + 1] - self.rowptr[rp]) / 2):
                start = self.colptr[self.rowptr[rp] + 2*c]
                end = self.colptr[self.rowptr[rp] + 2*c + 1]

                # Cython does not yet support "yield"
                out.append((r, start, end))

        return out

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
        cdef int i

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

    def set_start_row(self, start_row):
        """Set the first row where values occur.

        """
        if start_row <= (self._shape[0] - len(self.rowptr) + 1):
            self._start_row = start_row
        else:
            raise ValueError("Start row is too large for the current "
                             "shape.  Reshape the connectedregion first.")

    def get_start_row(self):
        return self._start_row

    start_row = property(fset=set_start_row, fget=get_start_row)

    def inside_boundary(self):
        """Return the indices for the inside boundary.

        """
        cdef int r, rp, c, start, end
        cdef list x = [], y = []

        for r, start, end in self._iterate_rows():
            x.append(start)
            y.append(r + self.start_row)

            end -= 1
            if end != start:
                x.append(end)
                y.append(r + self.start_row)

        return x, y

    def outside_boundary(self):
        pass

    def __add__(self, a):
        """Merge two regions.

        """
        raise NotImplementedError
