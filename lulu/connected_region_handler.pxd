# -*- python -*-

from connected_region cimport ConnectedRegion

cdef _iterate_rows(ConnectedRegion cr)
cpdef int nnz(ConnectedRegion cr)

cpdef get_shape(ConnectedRegion cr)
cdef _minimum_shape(ConnectedRegion cr)
cpdef reshape(ConnectedRegion cr, shape=?)
cpdef ConnectedRegion copy(ConnectedRegion cr)
cpdef set_start_row(ConnectedRegion cr, int start_row)
cpdef int get_start_row(ConnectedRegion cr)
cpdef int contains(ConnectedRegion cr, int r, int c)
cpdef outside_boundary(ConnectedRegion cr)
cpdef set_value(ConnectedRegion cr, int v)
cpdef int get_value(ConnectedRegion cr)
cpdef validate(ConnectedRegion cr)

# These methods are needed by the lulu decomposition to build
# connected regions incrementally

cpdef _new_row(ConnectedRegion cr)
cpdef _append_colptr(ConnectedRegion cr, int a, int b)
cpdef int _current_row(ConnectedRegion cr)

