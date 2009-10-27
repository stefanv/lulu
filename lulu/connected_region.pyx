# -*- python -*-
"""
Notes on file structure: we have a minimal data class,
ConnectedRegion, that stores the row and column pointers, size and
value for the ConnectedRegion.  The functions in
connected_region_handler do all the work on ConnectedRegions.  This
somewhat roundabout way of doing things is to ensure that the memory
size of ConnectedRegion is minimal, since we have to store thousands
of these objects.

"""

__all__ = ['ConnectedRegion']

import lulu.connected_region_handler as crh

cdef class ConnectedRegion:
    """
    A 4-connected region is stored in a modified Compressed
    Sparse Row matrix format.

    Since the region is connected, we only have to store one value.
    Along a single row, connected regions are stored as index pairs, e.g.

    ---00-000-- would be represented as [3, 5, 6, 9]

    This class should be queried using the methods in
    `connected_region_handler`.

    Attributes
    ----------
    rowptr : list of int
        `rowptr[i]` tells us where in `colptr` the elements of row i are
        described
    colptr : list of int
        Always contains 2N elements, where N are the number of connected
        regions (see description above).  Each entry describes the half-open
        interval ``(start_position, end_position]``.

    Parameters
    ----------
    shape : tuple
        Shape of the region.

    Optional Parameters
    -------------------
    value : int
        Region value.
    start_row : int
        First row in which values occur.
    rowptr, colptr : list of int
        See "Attributes".

    """

    # all class variables and their types are defined in connected_region.pxd

    def __init__(self, tuple shape, int value=0, int start_row=0,
                 list rowptr=None, list colptr=None):
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

        self._start_row = start_row

        # Initialise nnz (nr of non-zeros or area of region)
        cdef int i, n
        n = 0
        if len(self.rowptr) != 0:
            for i in range((self.rowptr[-1] - self.rowptr[0]) / 2):
                n += self.colptr[2*i + 1] - self.colptr[2*i]

        self._nnz = n
