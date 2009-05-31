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

cdef inline bool gt(int a, int b):
    return a > b

cdef inline bool lt(int a, int b):
    return a < b

cdef int _boundary_extremum(ConnectedRegion cr, int* img,
                            int max_rows, int max_cols,
                            bool (*func)(int, int),
                            int initial_extremum = 0):
    """Determine the extremal value on the boundary of a
    ConnectedRegion.

    Parameters
    ----------
    cr : ConnectedRegion
    img : Input image as positive integer array
    max_rows, max_cols : int
        Dimensions of img.
    func : callable
        Function used to test for the extreme value:

        func(img[r, c], cr.value)
    initial_extremum : int

    """
    cdef int i, r, c
    cdef int img_val
    cdef int extremum = initial_extremum

    cdef list x, y

    y, x = outside_boundary(cr)

    for i in range(1, len(y)):
        r = y[i]
        c = x[i]

        if (r < 0 or r >= max_rows) or (c < 0 or c >= max_cols):
            continue

        img_val = img[r*max_cols + c]
        if func(img_val, extremum):
            extremum = img_val

    return extremum

cdef int _boundary_maximum(ConnectedRegion cr, int* img,
                           int max_rows, int max_cols):
    return _boundary_extremum(cr, img, max_rows, max_cols, gt, -1)

cdef int _boundary_minimum(ConnectedRegion cr, int* img,
                           int max_rows, int max_cols):
    return _boundary_extremum(cr, img, max_rows, max_cols, lt, 256)

def boundary_maximum(ConnectedRegion cr, np.ndarray[np.int_t, ndim=2] img):
    return _boundary_maximum(cr, <int*>img.data, img.shape[0], img.shape[1])

def boundary_minimum(ConnectedRegion cr, np.ndarray[np.int_t, ndim=2] img):
    return _boundary_minimum(cr, <int*>img.data, img.shape[0], img.shape[1])


cdef inline min2(int a, int b):
    if a < b:
        return a
    else:
        return b

cdef inline max2(int a, int b):
    if a > b:
        return a
    else:
        return b

cpdef merge(ConnectedRegion a, ConnectedRegion b):
    """Merge b into a.  b and a must be connected.

    """
    cdef int r, rpt, cpt
    cdef int start_row = min2(a._start_row, b._start_row)
    cdef int end_row = max2(a._start_row + len(a.rowptr) - 2,
                            b._start_row + len(b.rowptr) - 2)

    cdef list new_colptr = []
    cdef list new_rowptr = []

    cdef list cols, merged_cols

    for r in range(start_row, end_row + 1):
        new_rowptr.append(len(new_colptr))

        # Non-overlapping, use b
        if r < a._start_row or r > (len(a.rowptr) + a._start_row - 2):
            rpt = r - b._start_row
            for i in range(b.rowptr[rpt], b.rowptr[rpt + 1]):
                new_colptr.append(b.colptr[i])

        # Non-overlapping, use a
        if r < b._start_row or r > (len(b.rowptr) + b._start_row - 2):
            rpt = r - a._start_row
            for i in range(a.rowptr[rpt], a.rowptr[rpt + 1]):
                new_colptr.append(a.colptr[i])

        # Overlapping: merge
        else:

            cols = []
            merged_cols = []
            rpt = r - a._start_row
            for i in range((a.rowptr[rpt + 1] - a.rowptr[rpt]) / 2):
                cpt = <int>a.rowptr[rpt] + 2 * i

                cols.append([a.colptr[cpt], a.colptr[cpt + 1]])

            rpt = r - b._start_row
            for i in range((b.rowptr[rpt + 1] - b.rowptr[rpt]) / 2):
                cpt = <int>b.rowptr[rpt] + 2 * i

                cols.append([b.colptr[cpt], b.colptr[cpt + 1]])

            cols.sort()

            for i in range(len(cols) - 1):
                if cols[i][1] == cols[i+1][0]:
                    cols[i+1][0] = cols[i][0]
                    continue
                merged_cols.append(cols[i])

            merged_cols.extend(cols[-1])

            new_colptr.extend(merged_cols)

    new_rowptr.append(len(new_colptr))

    a.colptr = new_colptr
    a.rowptr = new_rowptr
    a._start_row = start_row
    reshape(a)

cdef _set_array(int* arr, int rows, int cols,
               ConnectedRegion c, int value):
    """Set the value of arr to value over the connected region.

    """
    pass

def set_array(np.ndarray[np.int_t, ndim=2] arr,
              ConnectedRegion c, int value):
    return _set_array(<int*>arr.data, arr.shape[0], arr.shape[1],
                      c, value)

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
