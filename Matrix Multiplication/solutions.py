
from copy import deepcopy

diag = lambda x, n: [[x if i == j else 0 for i in range(n)] for j in range(n)]

def matmul(M1, M2):
    out = [[sum(a * M2[i][j] for i, a in enumerate(R1)) for j in range(len(M1))] for R1 in M1]
    return(out)

def matpow_naive(A, p):
    """
    Raise a matrix to a non-negative power.

    Parameters
    ----------
    A : list[list[int]]
        the matrix
    p : int
        the power

    Returns
    -------
    list[list[int]]
        the exponentiated matrix
    """
    if p == 0:
        return diag(1, len(A))
    B = deepcopy(A)
    for i in range(p - 1):
        B = matmul(B, A)
    return B


def matpow_dac(A, p):
    """
    Raise a matrix to a non-negative power.

    Parameters
    ----------
    A : list[list[int]]
        the matrix
    p : int
        the power

    Returns
    -------
    list[list[int]]
        the exponentiated matrix
    """
    if p == 0:
        return diag(1, len(A))
    elif p == 1:
        return A

    P = matpow_dac(A, p // 2)
    if p % 2 == 0:
        return matmul(P, P)
    else:
        return matmul(A, matmul(P, P))