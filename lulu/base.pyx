# -*- python -*-

__all__ = ['connected_regions', 'decompose']

import numpy as np

cimport numpy as np

import cython

from lulu.ccomp import label
from lulu.connected_region cimport ConnectedRegion
cimport lulu.connected_region_handler as crh

def connected_regions(np.ndarray[np.int_t, ndim=2] img):
    """Return ConnectedRegions that, together, compose the whole image.

    Parameters
    ----------
    img : ndarray
        Input image.

    Returns
    -------
    labels : ndarray
        `img`, labeled by connectivity.
    c : dict
        Dictionary of ConnectedRegions, indexed by label value.

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
                if crh._current_row(cur_region) != r:
                    crh._new_row(cur_region)

                # Add connected region
                crh._append_colptr(cur_region, connect_from, c)

                connect_from = c

    # finalise rows
    for cr in regions.itervalues():
        crh._new_row(cr)

    return labels, regions

cdef _reset_and_merge(ConnectedRegion cr, int new_value, dict regions,
                      int* image, int* labels, int rows, int cols):
    """
    Reset the value of a ConnectedRegion and handle all ensuing merges.

    """

    cdef list regions_to_merge = []
    cdef list labels_to_merge = []
    cdef list x, y
    cdef int i, r, c, idx
    cdef ConnectedRegion cur_region, tmp_region
    cdef int primary_label, cur_label, cr_label

    cr_label = labels[cr._start_row * cols + <int>cr.colptr[0]]
    primary_label = cr_label

    # Check boundary for regions to merge
    y, x = crh.outside_boundary(cr)

    for i in range(len(y)):
        r = y[i]
        c = x[i]

        idx = r*cols + c

        # boundary check
        if (r < 0 or r >= rows) or (c < 0 or c >= cols):
            continue

        if image[idx] == new_value:
            cur_label = labels[idx]

            # Don't process region twice
            if cur_label in labels_to_merge:
                continue

            cur_region = regions[cur_label]

            if cur_label < primary_label:
                # The current label occurs earlier on in the image than
                # cr.  This should then be the label for the merged region.
                primary_label = cur_label

            regions_to_merge.append(cur_region)
            labels_to_merge.append(cur_label)

            # Since this region is about to be merged, remove it from the
            # regions list
            del regions[cur_label]

    # Merge boundary regions, update regions, update labels and image

    cr._value = new_value
    for cur_region in regions_to_merge:
        # Perform merge
        crh.merge(cr, cur_region)

    # Update labels and image values
    crh._set_array(labels, rows, cols, cr, primary_label)
    crh._set_array(image, rows, cols, cr, new_value)

    # Place merged region in `regions`
    del regions[cr_label]
    regions[primary_label] = cr

def decompose(np.ndarray[np.int_t, ndim=2] img):
    cdef np.ndarray[np.int_t, ndim=2] labels
    cdef dict regions

    cdef ConnectedRegion cr
    cdef int nz

    cdef int* img_data = <int*>img.data
    cdef int max_rows = img.shape[0]
    cdef int max_cols = img.shape[1]

    labels, regions = connected_regions(img)
    cdef int* labels_data = <int*>labels.data

    cdef bool merged = True
    cdef int b_min, b_max

    for area in range(1000):
        # Examine regions of a certain size only
        for cr in regions.itervalues():
            nz = crh.nnz(cr)
            if nz != area:
                # Only interested in regions of a certain area.
                # This also filters out regions that have been merged already.
                continue

            b_max = crh._boundary_maximum(cr, img_data, max_cols, max_rows)

            # Do we have a maximal set?
            if b_max < cr._value:
                _reset_and_merge(cr, b_max, regions,
                                 img_data, labels_data, max_rows, max_cols)

        for cr in regions.itervalues():
            nz = crh.nnz(cr)
            if nz != area:
                # Only interested in regions of a certain area.
                # This also filters out regions that have been merged already.
                continue

            b_min = crh._boundary_minimum(cr, img_data, max_cols, max_rows)

            # Do we have a minimal set?
            if b_min > cr._value:
                _reset_and_merge(cr, b_min, regions,
                                 img_data, labels_data, max_rows, max_cols)

        merged = False

    return labels
