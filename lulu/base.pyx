__all__ = ['connected_regions']

import numpy as np

cimport numpy as np

import cython
from python_dict cimport PyDict_Next
ctypedef void PyObject

from lulu.ccomp import label
from lulu.connected_region cimport ConnectedRegion

def connected_regions(np.ndarray[np.int_t, ndim=2] img):
    """Return ConnectedRegions that, together, compose the whole image.

    """
    cdef int rows = img.shape[0]
    cdef int columns = img.shape[1]

    cdef ConnectedRegion cr

    # perform initial labeling
    cdef np.ndarray[np.int_t, ndim=2] labels = label(img)

    cdef dict regions = {}

    # create the first level components
    cdef int r = 0, c = 0, connect_from, prev_label = 0, cur_label = 0

    for r in range(rows):
        connect_from = 0

        for c in range(1, columns + 1):
            prev_label = labels[r, c - 1]

            if c < columns:
                cur_label = labels[r, c]
            else:
                cur_label = prev_label

            # Different region?
            if prev_label != cur_label or c == columns:

                # New region?
                if prev_label > len(regions) - 1:
                    regions[prev_label] = ConnectedRegion(
                        shape=(rows, columns),
                        value=img[r, connect_from],
                        start_row=r,
                        rowptr=[0])

                cur_region = regions[prev_label]

                # New row?
                if cur_region._current_row() != r:
                    cur_region._new_row()

                # Add connected region
                cur_region._append_colptr(connect_from, c)

                connect_from = c

    # finalise rows
    for cr in regions.itervalues():
        cr._new_row()

    return regions
