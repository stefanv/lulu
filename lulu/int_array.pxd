cdef class IntArray:
    cdef int* buf
    cdef int cap
    cdef int size

cpdef inline append(IntArray arr, int)
cpdef int max(IntArray)
cpdef int min(IntArray)
cpdef copy(IntArray, IntArray)
cpdef from_list(IntArray, list)
cpdef int get(IntArray, int)
cpdef list to_list(IntArray arr)
