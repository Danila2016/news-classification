""" Intersection kernel matrix precomputation for scipy.sparse.csr_matrices """

cimport numpy as cnp

def kernel(int na, int nb,
    cnp.ndarray[int, ndim=1] indptra, cnp.ndarray[int, ndim=1] indptrb,
    cnp.ndarray[int, ndim=1] indicesa, cnp.ndarray[int, ndim=1] indicesb,
    cnp.ndarray[cnp.float64_t, ndim=1] dataa, cnp.ndarray[cnp.float64_t, ndim=1] datab,
    cnp.ndarray[cnp.float64_t, ndim=2] K):

    cdef int i, j, I, J, IEND, JEND

    for i in range(na):
        for j in range(nb):
            I = indptra[i]
            J = indptrb[j]
            IEND = indptra[i+1]
            JEND = indptrb[j+1]
            while (I < IEND or J < JEND):
                if (I != IEND and (J == JEND or indicesa[I] < indicesb[J])):
                    I += 1
                elif (I == IEND or indicesa[I] > indicesb[J]):
                    J += 1
                else:
                    if (dataa[I] < datab[J]):
                        K[i,j] += dataa[I]
                    else:
                        K[i,j] += datab[J]
                    
                    I += 1
                    J += 1
