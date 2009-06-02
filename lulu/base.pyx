# -*- python -*-

__all__ = ['connected_regions', 'decompose']

import numpy as np

cimport numpy as np

import cython
import sys

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

cdef list _identify_pulses_and_merges(dict regions, int area, dict pulses,
                                      int* img_data, int rows, int cols,
                                      mode=0):
    """Return pulses that need to be removed, as well as the
    positions of areas that requires merging after the removal.

    Parameters
    ----------
    mode : int
        0 - U (upper), raise minima
        1 - L (lower), lower maxima
        2 - B (both), do both

    """
    cdef ConnectedRegion cr, cr_save

    cdef int b_max = cr._value
    cdef int b_min = cr._value

    cdef list merge_region_positions = []

    if area not in pulses:
        pulses[area] = []

    # Examine regions of a certain size only
    for cr in regions.itervalues():
        if cr._nnz != area:
            # Only interested in regions of a certain area.
            continue

        old_value = cr._value

        # Upper
        if mode == 0 or mode == 2:
            b_min = crh._boundary_minimum(cr, img_data, rows, cols)

            # Minimal set
            if b_min > old_value:
                cr._value = b_min

        # Lower
        if mode == 1 or mode == 2:
            b_max = crh._boundary_maximum(cr, img_data, rows, cols)

            # Maximal set
            if b_max < old_value:
                cr._value = b_max

        # Minimal or maximal region detected
        if cr._value != old_value:
            crh._set_array(img_data, rows, cols, cr, cr._value)

            cr_save = crh.copy(cr)
            cr_save._value = old_value - cr._value # == pulse height
            pulses[area].append(cr_save)

            # We don't need to re-examine each merged area for re-merging,
            # so can still optimise this later.
            merge_region_positions.append((cr._start_row, <int>cr.colptr[0]))

    if len(pulses[area]) == 0:
        del pulses[area]

    return merge_region_positions

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

    print "[> 0%% %s ]" % (" "*50),
    sys.stdout.flush()
    for area in range(levels):
        percentage = area*100/levels
        if percentage != percentage_done:
            print "\r[=%s> %d%%" % ("=" * (percentage/2), percentage),
            sys.stdout.flush()
            percentage_done = percentage

        try:
            if area_histogram[area] <= 0:
                continue
        except KeyError:
            continue

        # Upper
        merge_region_positions = \
            _identify_pulses_and_merges(regions, area, pulses,
                                        img_data, max_rows, max_cols, 0)

        _merge_all(merge_region_positions,
                   regions, area_histogram,
                   img_data, labels_data, max_rows, max_cols)

        # Lower
        merge_region_positions = \
            _identify_pulses_and_merges(regions, area, pulses,
                                        img_data, max_rows, max_cols, 1)

        _merge_all(merge_region_positions,
                   regions, area_histogram,
                   img_data, labels_data, max_rows, max_cols)


    print
    return pulses
