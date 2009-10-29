# -*- python -*-

cimport int_array
cimport stdlib

cdef class IntArray:
    """See int_array.pxd for members.

    """
    def __cinit__(self):
        self.cap = 6
        self.buf = <int*>stdlib.malloc(sizeof(int) * self.cap) # storage
        self.size = 0

    def __dealloc__(self):
        stdlib.free(self.buf)

cpdef inline append(IntArray arr, int x):
    cdef int* new_buf
    cdef int i

    if arr.size == arr.cap:
        # Array is full -- allocate new memory
        arr.cap = arr.cap * 2
        new_buf = <int*>stdlib.malloc(sizeof(int) * arr.cap)

        for i in range(arr.size):
            new_buf[i] = arr.buf[i]

        stdlib.free(arr.buf)
        arr.buf = new_buf

    arr.buf[arr.size] = x
    arr.size += 1

cpdef int max(IntArray arr):
    cdef int i,  m = arr.buf[0]
    for i in range(1, arr.size):
        if arr.buf[i] > m:
            m = arr.buf[i]

    return m

cpdef int min(IntArray arr):
    cdef int i, m = arr.buf[0]
    for i in range(1, arr.size):
        if arr.buf[i] < m:
            m = arr.buf[i]

    return m

cpdef copy(IntArray src, IntArray dst):
    cdef int i

    dst.size = 0
    for i in range(src.size):
        append(dst, src.buf[i])

cpdef from_list(IntArray arr, list ii):
    if ii is not None:
        arr.size = 0
        for i in ii:
            append(arr, i)

cpdef int get(IntArray arr, int idx):
    return arr.buf[idx]

cpdef list to_list(IntArray arr):
    cdef list out = []
    cdef int i

    for i in range(arr.size):
        out.append(arr.buf[i])

    return out
