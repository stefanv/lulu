cdef class ConnectedRegion:
    cdef int _value
    cdef int _start_row
    cdef list rowptr, colptr
    cdef int _nnz
    cdef tuple _shape

    cdef _iterate_rows(self)
    cpdef int nnz(self)

    cpdef get_shape(self)
    cdef _minimum_shape(self)
    cpdef reshape(self, shape=?)
    cpdef ConnectedRegion copy(self)
    cpdef set_start_row(self, start_row)
    cpdef int get_start_row(self)
    cpdef int contains(self, int r, int c)
    cpdef outside_boundary(self)
    cpdef set_value(self, int v)
    cpdef int get_value(self)
    cpdef validate(self)

    # These methods are needed by the lulu decomposition to build
    # connected regions incrementally

    cpdef _new_row(self)
    cpdef _append_colptr(self, int a, int b)
    cpdef int _current_row(self)
