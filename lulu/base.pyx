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
        cr._nnz = crh.nnz(cr)

    return labels, regions

cdef _merge_all(list merge_region_positions, dict regions, dict area_histogram,
                int* image, int* labels, int rows, int cols):
    """
    Merge all regions that have connections on their boundaries.

    """
    cdef list x, y
    cdef int i, p_row, p_col, r, c, idx, new_area
    cdef ConnectedRegion cr, this_region
    cdef int this_label, cr_label, new_label

    for p_row, p_col in merge_region_positions:
        cr_label = labels[p_row*cols + p_col]
        cr = regions[cr_label]

        y, x = crh.outside_boundary(cr)

        # Check boundary for regions to merge
        for i in range(len(y)):
            r = y[i]
            c = x[i]

            # boundary check
            if (r < 0 or r >= rows) or (c < 0 or c >= cols):
                continue

            idx = r*cols + c

            # Check whether these regions should be merged (only if they
            # haven't been already)
            this_label = labels[idx]
            if image[idx] == cr._value and this_label != cr_label:
                this_region = regions[this_label]

                # Merge; update regions, labels and image
                new_label = crh.min2(cr_label, this_label)

                # Update area histogram
                new_area = cr._nnz + this_region._nnz
                area_histogram[cr._nnz] -= 1
                area_histogram[this_region._nnz] -= 1
                try:
                    area_histogram[new_area] += 1
                except KeyError:
                    area_histogram[new_area] = 1

                crh.merge(cr, this_region)

                # OPT: Only one of these deletes need to be executed.
                del regions[cr_label]
                del regions[this_label]
                regions[new_label] = cr
                cr_label = new_label

                # Update labels
                crh._set_array(labels, rows, cols, cr, new_label)
                cr._nnz = new_area

def decompose(np.ndarray[np.int_t, ndim=2] img):
    cdef np.ndarray[np.int_t, ndim=2] labels
    cdef dict regions

    cdef ConnectedRegion cr, cr_save
    cdef int nz

    cdef int* img_data = <int*>img.data
    cdef int max_rows = img.shape[0]
    cdef int max_cols = img.shape[1]

    labels, regions = connected_regions(img)
    cdef int* labels_data = <int*>labels.data

    cdef int b_min, b_max
    cdef list merge_region_positions

    cdef dict area_histogram = {}
    cdef dict pulses = {}

    cdef int old_value, levels, percentage_done, percentage

    for cr in regions.itervalues():
        try:
            area_histogram[cr._nnz] += 1
        except KeyError:
            area_histogram[cr._nnz] = 1

    levels = max_cols * max_rows + 1
    percentage_done = 0
    for area in range(levels):
        percentage = area*100/levels
        if percentage % 10 == 0 and percentage != percentage_done:
            print "%s%%" % percentage
            percentage_done = percentage

        try:
            if area_histogram[area] <= 0:
                continue
        except KeyError:
            continue

        merge_region_positions = []

        # Examine regions of a certain size only
        for cr in regions.itervalues():
            if cr._nnz != area:
                # Only interested in regions of a certain area.
                continue

            # Could combine these two functions calls, then we need
            # only one loop.
            b_max = crh._boundary_maximum(cr, img_data, max_rows, max_cols)
            b_min = crh._boundary_minimum(cr, img_data, max_rows, max_cols)

            # Do we have a maximal or minimal set?
            old_value = cr._value
            if b_max < cr._value:
                # Drop peak
                cr._value = b_max
                crh._set_array(img_data, max_rows, max_cols, cr, cr._value)
            elif b_min > cr._value:
                # Raise trough
                cr._value = b_min
                crh._set_array(img_data, max_rows, max_cols, cr, cr._value)
            else:
                continue

            if area not in pulses:
                pulses[area] = []
            cr_save = crh.copy(cr)
            cr_save._value = old_value - cr._value # == pulse height
            pulses[area].append(cr_save)

            # We don't need to re-examine each merged area for re-merging,
            # so can still optimise this later.
            merge_region_positions.append((cr._start_row, <int>cr.colptr[0]))

        _merge_all(merge_region_positions, regions, area_histogram,
                   img_data, labels_data, max_rows, max_cols)

    return pulses
