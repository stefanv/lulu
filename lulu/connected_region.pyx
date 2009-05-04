import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport stdlib

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
    cdef int _value
    cdef int _start_row
    cdef list rowptr, colptr
    cdef int _nnz
    cdef tuple _shape

    def __init__(self, shape, int value=0, int start_row=0,
                 rowptr=None, colptr=None):
        if shape is None:
            raise ValueError("Shape must be specified.")

        self._shape = shape
        self._value = value

        if rowptr is None:
            self.rowptr = []
        else:
            self.rowptr = rowptr

        if colptr is None:
            self.colptr = []
        else:
            self.colptr = colptr

        self.start_row = start_row

    def _iterate_rows(self):
        """For each row, return the connected columns as

        (row, start, end)

        Note that this row includes the row offset.

        """
        cdef list out = []
        cdef int r, c, start, end

        for r in range(len(self.rowptr) - 1):
            for c in range((self.rowptr[r + 1] - self.rowptr[r]) / 2):
                start = self.colptr[self.rowptr[r] + 2*c]
                end = self.colptr[self.rowptr[r] + 2*c + 1]

                # Cython does not yet support "yield"
                out.append((r + self.start_row, start, end))

        return out

    @cython.boundscheck(True)
    def todense(self):
        """Convert the connected region to a dense array.

        """
        self.validate()

        shape = self._shape
        if shape is None:
            shape = self._minimum_shape()

        cdef np.ndarray[np.int_t, ndim=2] out = np.zeros(shape, dtype=np.int)

        cdef int row, start, end

        for row, start, end in self._iterate_rows():
            for k in range(start, end):
                out[row, k] = self.value

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

    def get_shape(self):
        return self._shape

    shape = property(fget=get_shape, fset=reshape)

    def _minimum_shape(self):
        """Return the minimum shape into which the connected region can fit.

        """
        return (self.start_row + len(self.rowptr) - 1, max(self.colptr))

    def reshape(self, shape=None):
        """Set the shape of the connected region.

        Useful when converting to dense.
        """
        if shape is None:
            self._shape = self._minimum_shape()
        elif (shape >= self._minimum_shape()):
            self._shape = shape
        else:
            raise ValueError("Minimum shape is %s." % self._minimum_shape())

    def copy(self):
        """Return a deep copy of the connected region.

        """
        return ConnectedRegion(shape=self._shape, value=self.value,
                               start_row=self.start_row,
                               rowptr=list(self.rowptr),
                               colptr=list(self.colptr))

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

    def contains(self, int r, int c):
        """Does the connected region contain and element at (r, c)?

        """
        cdef i

        r -= self.start_row

        rows = len(self.rowptr)

        if r < 0 or r > rows - 2:
            return False

        if c < 0 or c >= self._shape[1]:
            return False

        for i in range((self.rowptr[r + 1] - self.rowptr[r]) / 2):
            if (c >= self.colptr[self.rowptr[r] + 2*i]) and \
               (c < self.colptr[self.rowptr[r] + 2*i + 1]):
                return True

        return False

    def inside_boundary(self):
        """Calculate the inside boundary.

        Returns
        -------
        r : list of int, length N
            Row positions.
        c : list of int, length N
            Column positions.

        """
        cdef int row, start, end
        cdef list x = [], y = []

        for row, start, end in self._iterate_rows():
            x.append(start)
            y.append(row)

            end -= 1
            if end != start:
                x.append(end)
                y.append(row)

        return y, x

    def outside_boundary(self):
        """Calculate the outside boundary using a scanline approach.

        Notes
        -----
        A scanline is constructed that is as wide as the region.  The
        scanline moves down the image from the top.  For each position
        in the scanline, if

        - the pixel above or below is part of the region
        - and the pixel at the current position is not

        then assign this position as part of the outside boundary.

        The advantage of this approach is that we are guaranteed the
        boundary positions ordered from top left to bottom right,
        which will be useful later when we join two regions together.

        Theoretically, an interval tree should be more efficient to
        determine the overlap between connected intervals, but the
        data structures and allocations required are more complex.  We
        shall have to benchmark both to know for sure.

        As an optimisation, we evaluate only points next to inside
        boundary positions.

        """
        cdef int i # scanline row-position
        cdef int j # column position in scanline
        cdef int start, end, k
        cdef list x = [], y = []

        cdef int columns = self._shape[1]
        cdef int rows = len(self.rowptr) - 1

        cdef int scanline_size = sizeof(int) * columns
        cdef int* line_above = <int*>stdlib.malloc(scanline_size)
        cdef int* line = <int*>stdlib.malloc(scanline_size)
        cdef int* line_below = <int*>stdlib.malloc(scanline_size)

        for j in range(columns):
            line[j] = 0
            line_below[j] = 0

        for i in range(rows + 2):
            # Update scanline and line above scanline
            for j in range(columns):
                line_above[j] = line[j]
                line[j] = line_below[j]

            # When the scanline reaches the last line,
            # fill line_below with zeros
            if i <= rows:
                for j in range(columns):
                    line_below[j] = 0

            # Update line below scanline
            if i < rows:
                for j in range((self.rowptr[i + 1] - self.rowptr[i]) / 2):
                    start = self.colptr[self.rowptr[i] + 2*j]
                    end = self.colptr[self.rowptr[i] + 2*j + 1]

                    for k in range(start, end):
                        line_below[k] = 1

            for j in range(columns):
                # Test four neighbours for connections
                if j == 0 and line[j] == 1:
                    x.append(-1)
                    y.append(i - 1 + self.start_row)

                if (line[j] == 0) and \
                   (line_above[j] == 1 or line_below[j] == 1 or
                    ((j - 1) >= 0 and line[j - 1] == 1) or \
                    ((j + 1) < columns and line[j + 1] == 1)):
                    x.append(j)
                    y.append(i - 1 + self.start_row)

                if j == columns - 1 and line[j] == 1:
                    x.append(columns)
                    y.append(i - 1 + self.start_row)

        stdlib.free(line_above)
        stdlib.free(line)
        stdlib.free(line_below)

        return y, x

    def __add__(self, a):
        """Merge two regions.

        """
        raise NotImplementedError

    def set_value(self, v):
        self._value = v

    def get_value(self):
        return self._value

    value = property(fset=set_value, fget=get_value)

    def validate(self):
        if self.rowptr[-1] != len(self.colptr):
            raise RuntimeError("ConnectedRegion was not finalised.  Ensure "
                               "rowptr[-1] points beyond last entry of "
                               "colptr.")

        if len(self.colptr) % 2 != 0:
            raise RuntimeError("Colptr must have 2xN entries.")

