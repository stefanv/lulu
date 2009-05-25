# -*- python -*-

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cimport stdlib

cimport lulu.connected_region_handler as crh

# Cython does not make it particularly easy to cdef static methods,
# so we define these methods in their own module so they handle
# similarly to static methods.

cdef _iterate_rows(ConnectedRegion cr):
    """For each row, return the connected columns as

    (row, start, end)

    Note that this row includes the row offset.

    """
    cdef list rowptr = cr.rowptr
    cdef list colptr = cr.colptr

    cdef list out = []
    cdef int r, c, start, end
    for r in range(len(rowptr) - 1):
        for c in range((rowptr[r + 1] - rowptr[r]) / 2):
            start = colptr[<int>rowptr[r] + 2*c]
            end = colptr[<int>rowptr[r] + 2*c + 1]

            # Cython does not yet support "yield"
            out.append((r + cr._start_row, start, end))

    return out

cpdef int nnz(ConnectedRegion cr):
    """Return the number of non-zero elements.

    """
    cdef int n = 0
    cdef int i

    for i in range(len(cr.colptr) / 2):
        n += cr.colptr[2*i + 1] - cr.colptr[2*i]

    return n

cpdef get_shape(ConnectedRegion cr):
    return cr._shape

cdef _minimum_shape(ConnectedRegion cr):
    """Return the minimum shape into which the connected region can fit.

    """
    return (cr._start_row + len(cr.rowptr) - 1, max(cr.colptr))

cpdef reshape(ConnectedRegion cr, shape=None):
    """Set the shape of the connected region.

    Useful when converting to dense.
    """
    if shape is None:
        cr._shape = crh._minimum_shape(cr)
    elif (shape >= crh._minimum_shape(cr)):
        cr._shape = shape
    else:
        raise ValueError("Minimum shape is %s." % \
                         crh._minimum_shape(cr))

cpdef ConnectedRegion copy(ConnectedRegion cr):
    """Return a deep copy of the connected region.

    """
    return ConnectedRegion(shape=cr._shape, value=cr._value,
                           start_row=cr._start_row,
                           rowptr=list(cr.rowptr),
                           colptr=list(cr.colptr))

cpdef set_start_row(ConnectedRegion cr, int start_row):
    """Set the first row where values occur.

    """
    if start_row <= (cr._shape[0] - len(cr.rowptr) + 1):
        cr._start_row = start_row
    else:
        raise ValueError("Start row is too large for the current "
                         "shape.  Reshape the connectedregion first.")

cpdef int get_start_row(ConnectedRegion cr):
    return cr._start_row

cpdef int contains(ConnectedRegion cr, int r, int c):
    """Does the connected region contain and element at (r, c)?

    """
    cdef i

    r -= cr._start_row

    rows = len(cr.rowptr)

    if r < 0 or r > rows - 2:
        return False

    if c < 0 or c >= cr._shape[1]:
        return False

    for i in range((cr.rowptr[r + 1] - cr.rowptr[r]) / 2):
        if (c >= cr.colptr[cr.rowptr[r] + 2*i]) and \
           (c < cr.colptr[cr.rowptr[r] + 2*i + 1]):
            return True

    return False

cpdef outside_boundary(ConnectedRegion cr):
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

    cdef int rows = len(cr.rowptr) - 1

    cdef int col_min = min(cr.colptr)
    cdef int col_max = max(cr.colptr)
    cdef int columns = col_max - col_min

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
            for j in range((cr.rowptr[i + 1] - cr.rowptr[i]) / 2):
                start = cr.colptr[cr.rowptr[i] + 2*j]
                end = cr.colptr[cr.rowptr[i] + 2*j + 1]

                for k in range(start - col_min, end - col_min):
                    line_below[k] = 1

        for j in range(columns):
            # Test four neighbours for connections
            if j == 0 and line[j] == 1:
                x.append(-1 + col_min)
                y.append(i - 1 + cr._start_row)

            if (line[j] == 0) and \
               (line_above[j] == 1 or line_below[j] == 1 or
                ((j - 1) >= 0 and line[j - 1] == 1) or \
                ((j + 1) < columns and line[j + 1] == 1)):
                x.append(j + col_min)
                y.append(i - 1 + cr._start_row)

            if j == columns - 1 and line[j] == 1:
                x.append(columns + col_min)
                y.append(i - 1 + cr._start_row)

    stdlib.free(line_above)
    stdlib.free(line)
    stdlib.free(line_below)

    return y, x

cpdef set_value(ConnectedRegion cr, int v):
    cr._value = v

cpdef int get_value(ConnectedRegion cr):
    return cr._value


cpdef validate(ConnectedRegion cr):
    if cr.rowptr[-1] != len(cr.colptr):
        raise RuntimeError("ConnectedRegion was not finalised.  Ensure "
                           "rowptr[-1] points beyond last entry of "
                           "colptr.")

    if len(cr.colptr) % 2 != 0:
        raise RuntimeError("Colptr must have 2xN entries.")

# These methods are needed by the lulu decomposition to build
# connected regions incrementally

cpdef _new_row(ConnectedRegion cr):
    cdef int L = len(cr.colptr)

    if not cr.rowptr[-1] == L:
        cr.rowptr.append(L)

cpdef _append_colptr(ConnectedRegion cr, int a, int b):
    cr.colptr.append(a)
    cr.colptr.append(b)

cpdef int _current_row(ConnectedRegion cr):
    return len(cr.rowptr) + cr._start_row - 1


def todense(ConnectedRegion cr):
    """Convert the connected region to a dense array.

    """
    crh.validate(cr)

    shape = cr._shape
    if shape is None:
        shape = crh._minimum_shape(cr)

    cdef np.ndarray[np.int_t, ndim=2] out = np.zeros(shape, dtype=np.int)

    cdef int row, start, end

    for row, start, end in crh._iterate_rows(cr):
        for k in range(start, end):
            out[row, k] = cr._value

    return out