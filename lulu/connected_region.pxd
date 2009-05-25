# -*- python -*-

cdef class ConnectedRegion:
    cdef int _value
    cdef int _start_row
    cdef list rowptr, colptr
    cdef int _nnz
    cdef tuple _shape
