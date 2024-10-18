# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
from numpy import dot, pi, log, linalg, zeros, float
cimport numpy as np
from libc.stdio cimport printf

from cpython.bytes cimport PyBytes_FromStringAndSize

def unique_int_string(np.ndarray[np.int64_t, ndim=2] a):
    cdef int i, len_before
    cdef int nr = a.shape[0]
    cdef int nc = a.shape[1]
    cdef set s = set()
    cdef np.ndarray[np.uint8_t, cast = True] idx = np.zeros(nr, dtype='bool')
    cdef bytes string

    for i in range(nr):
        len_before = len(s)
        string = PyBytes_FromStringAndSize(<char*>&a[i, 0], sizeof(np.int64_t) * nc)
        s.add(string)
        if len(s) > len_before:
            idx[i] = True
    return idx

def sub2ind(array_shape, rows, cols):
    return rows + array_shape[0] * cols

from numpy cimport int64_t, uint8_t
import numpy as np

cdef extern from 'helper.h' nogil:
    cdef cppclass ArraySet:
        ArraySet()
        ArraySet(size_t)
        bint add(char*)


def unique_ints(int64_t[:, :] a):
    cdef:
        Py_ssize_t i, nr = a.shape[0], nc = a.shape[1]
        ArraySet s = ArraySet(sizeof(int64_t) * nc)
        uint8_t[:] idx = np.zeros(nr, dtype='uint8')

        bint found;

    for i in range(nr):
        found = s.add(<char*>&a[i, 0])
        if found:
            idx[i] = True
            printf("\n%d",i)
        else:
            print("\n DKM %d",i)
    return idx

DTYPE = float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def gibbswrap_jointpredupdt_custom(np.ndarray[double, ndim=2] P0, int m):
    cdef int n1
    cdef np.ndarray[int64_t, ndim=2] assignments
    cdef np.ndarray[double, ndim=1] costs
    cdef np.ndarray[int, ndim=1] currsoln
    cdef np.ndarray[double, ndim=1] tempsamp
    cdef np.ndarray[long long, ndim=1] idxold

    n1 = P0.shape[0];

    if m == 0:
        m = 1 # return at least one solution

    assignments = np.zeros((m, n1), dtype=np.int64);
    costs = np.zeros(m);

    currsoln = np.arange(n1, 2 * n1);  # use all missed detections as initial solution
    assignments[0, :] = currsoln;
    costs[0] = sum(P0.flatten('F')[sub2ind([P0.shape[0], P0.shape[1]], np.arange(0, n1), currsoln)]);
    for sol in range(1, m):
        for var in range(0, n1):
            tempsamp = np.exp(-P0[var, :]);  # grab row of costs for current association variable
            # lock out current and previous iteration step assignments except for the one in question
            tempsamp[np.delete(currsoln, var)] = 0;
            idxold = np.nonzero(tempsamp > 0)[0];
            tempsamp = tempsamp[idxold];
            currsoln[var] = np.digitize(np.random.rand(1), np.concatenate(([0], np.cumsum(tempsamp) / sum(tempsamp))));
            currsoln[var] = idxold[currsoln[var]-1];
        assignments[sol, :] = currsoln;
        costs[sol] = sum(P0.flatten('F')[sub2ind([P0.shape[0], P0.shape[1]], np.arange(0, n1), currsoln)]);
    index = unique_int_string(assignments) #np.unique(assignments, return_index=True, return_inverse=True, axis=0);
    assignments = assignments[index, :]
    costs = costs[index]

    return assignments, costs
