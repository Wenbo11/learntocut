cimport numpy as np
import numpy

def vector2toeplitz(np.ndarray a, np.ndarray b):
    # construct toeplitz matrix from vector a,b
    n = a.size 
    m = b.size +  1
    z = numpy.zeros([n,m])
    for i in range(n):
        for j in range(m):
            if i >= j:
                z[i,j] = a[i-j]
            else:
                z[i,j] = b[j-i-1]
    return z