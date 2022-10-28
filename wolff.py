import numpy as np
from numba import njit

@njit
def IsingUpdate(l,betaJ,h,nSweeps):
    '''

    Parameters
    ----------
    l : TYPE
        DESCRIPTION.
    betaJ : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    nSweeps : TYPE
        DESCRIPTION.

    Returnss
    -------
    None.

    '''
    N = l.shape[0]
    # Simulate
    for t in range((N**2)*nSweeps):
        i, j = np.random.rand(2)
        s = l[i,j]
        # Calculate neighbor sum
        neighs = 0
        for ip in [i-1,i+1]:
            for jp in [j-1,j+1]:
                neighs += l[ip%N,jp%N]
        # Calculate energy difference
        betaDeltaE = betaJ*neighs*s
        # Evaluate flip probability
        if betaDeltaE > np.random.rand():
            l[i,j] = -l[i,j]
    return l