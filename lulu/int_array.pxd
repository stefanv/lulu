cdef class IntArray:
    cdef int* buf
    cdef int cap
    cdef int size

cpdef append(IntArray arr, int)
cpdef inline max(IntArray)
cpdef inline min(IntArray)
cpdef copy(IntArray, IntArray)
cpdef from_list(IntArray, list)
cpdef get(IntArray, int)
