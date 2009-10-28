# -*- python -*-

__all__ = ['connected_regions', 'decompose', 'reconstruct']

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

            # Different region or end of row reached
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

cdef _merge_all(dict merges, dict regions, dict area_histogram,
                np.int_t* labels, int rows, int cols):
    """
    Merge all regions that have connections on their boundaries.

    """
    cdef ConnectedRegion cr_a, cr_b
    cdef int label, new_area, b_label

    for label in merges:
        try:
            cr_a = regions[label]
        except KeyError:
            continue

        for cr_b in merges[label]:
            b_label = cr_b._start_row * cols + <int>cr_b.colptr[0]
            if b_label == label:
                # Regions have alreay been merged
                continue

            # Merge; update regions, labels
            # Image has already been updated in identify_pulses_and_merges

            # Update area histogram
            new_area = cr_a._nnz + cr_b._nnz
            area_histogram[cr_a._nnz] -= 1
            area_histogram[cr_b._nnz] -= 1
            try:
                area_histogram[new_area] += 1
            except KeyError:
                area_histogram[new_area] = 1

            # Can delete regions later if needed; check speed implications
#            del regions[labels[b_label]]
            crh._set_array(labels, rows, cols, cr_b, label)

            # Update labels of cr_b
            crh.merge(cr_a, cr_b) # merge b into a

cdef dict _identify_pulses_and_merges(dict regions, int area, dict pulses,
                                      np.int_t* img_data, np.int_t* labels,
                                      int rows, int cols, int mode=0):
    """Return positions of areas that need to be merged after the removal.

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
    cdef int old_value

    cdef dict merges = {}
    cdef list y, x
    cdef int i, label, idx
    cdef int xi, yi

    if area not in pulses:
        pulses[area] = []

    # Examine regions of a certain size only
    for label, cr in regions.iteritems():
        if cr._nnz != area:
            # Only interested in regions of a certain area.
            continue

        old_value = cr._value

        y, x = crh.outside_boundary(cr)

        # Upper
        if mode == 0 or mode == 2:
            b_min = crh._boundary_minimum(x, y, img_data, rows, cols)

            # Minimal set
            if b_min > old_value:
                cr._value = b_min

        # Lower
        if mode == 1 or mode == 2:
            b_max = crh._boundary_maximum(x, y, img_data, rows, cols)

            # Maximal set
            if b_max < old_value:
                cr._value = b_max

        # Minimal or maximal region detected
        if cr._value != old_value:
            # This should occur exactly once: on the last
            # pulse that covers the whole image
            if cr._value == -1:
                cr._value = 0

            crh._set_array(img_data, rows, cols, cr, cr._value)
            try:
                # Check if this key exists
                merges[label].discard(0)
            except KeyError:
                merges[label] = set([])

            cr_save = crh.copy(cr)
            cr_save._value = old_value - cr._value # == pulse height
            pulses[area].append(cr_save)

            for i in range(len(x)):
                xi = x[i]
                yi = y[i]

                if (xi < 0) or (xi >= cols) or (yi < 0) or (yi >= rows):
                    # Position outside boundary
                    continue

                idx = yi * cols + xi

                if img_data[idx] == cr._value:
                    merges[label].add(regions[labels[idx]])

    if len(pulses[area]) == 0:
        del pulses[area]

    return merges

def decompose(np.ndarray[np.int_t, ndim=2] img):
    """Decompose a two-dimensional signal into pulses.

    Parameters
    ----------
    img : 2-D ndarray of ints
        Input signal.

    Returns
    -------
    pulses : dict
        Dictionary of ConnectedRegion objects, indexed by pulse area.

    See Also
    --------
    reconstruct

    """
    img = img.copy()

    cdef np.ndarray[np.int_t, ndim=2] labels
    cdef dict regions

    cdef ConnectedRegion cr
    cdef int nz

    cdef np.int_t* img_data = <np.int_t*>img.data
    cdef int max_rows = img.shape[0]
    cdef int max_cols = img.shape[1]

    # labels (array): `img`, numbered according to connected region
    # regions (dict): ConnectedRegions, indexed by label value.
    labels, regions = connected_regions(img)
    cdef np.int_t* labels_data = <np.int_t*>labels.data

    cdef set merge_region_positions

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
        merges = \
               _identify_pulses_and_merges(regions, area, pulses,
                                           img_data, labels_data,
                                           max_rows, max_cols, 0)

        _merge_all(merges, regions, area_histogram,
                   labels_data, max_rows, max_cols)

        # Lower
        merges = \
               _identify_pulses_and_merges(regions, area, pulses,
                                           img_data, labels_data,
                                           max_rows, max_cols, 1)

        _merge_all(merges, regions, area_histogram,
                   labels_data, max_rows, max_cols)


    print
    return pulses

def reconstruct(dict regions, tuple shape, int min_area=-1, int max_area=-1):
    """Reconstruct an image from the given connected regions / pulses.

    Parameters
    ----------
    regions : dict
        Impulses indexed by area.  This is the output of `decompose`.
    shape : tuple
        Shape of the output image.
    min_area, max_area : int
        Impulses with areas in [min_area, max_area] are used for the
        reconstruction.

    Returns
    -------
    out : ndimage
        Reconstructed image.
    areas : 1d ndarray
        Pulses with these areas occur in the image.
    area_count : 1d ndarray
        For each area in the above list, there are this many impulses.

    """
    cdef ConnectedRegion cr

    cdef np.ndarray[np.int_t, ndim=2] out = np.zeros(shape, dtype=int)

    if max_area == -1:
        max_area = out.shape[0] * out.shape[1] + 1

    if min_area == -1:
        min_area = 0

    cdef list areas = []
    cdef list area_count = []
    cdef int a

    for a in regions:
        if a >= min_area and a <= max_area:
            areas.append(a)
            area_count.append(0)

            area_count[-1] += len(regions[a])

            for cr in regions[a]:
                crh._set_array(<np.int_t*>out.data, out.shape[0], out.shape[1],
                               cr, cr._value, 1)

    areas_arr, area_count_arr = np.array(areas), np.array(area_count)

    # Sort by area
    ind = np.argsort(areas_arr)
    areas_arr = areas_arr[ind]
    area_count_arr = area_count_arr[ind]

    return out, areas, area_count_arr
