import pybryt
from solutions import diag, matmul
from copy import deepcopy

naive_annots = []

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
        d = diag(1, len(A))
        naive_annots.append(pybryt.Value(
            d,
            name="identity",
            success_message="Returned identity when p == 0",
            failure_message="Incorrect return value when p == 0",
        ))
        return d

    B = deepcopy(A)
    partial = pybryt.Value(
        B,
        name="partial-product",
        success_message="Found correct partial products",
        failure_message="Incorrect partial products",
    )

    for i in range(p - 1):
        B = matmul(B, A)
        next_partial = pybryt.Value(
            B,
            name="partial-product",
            success_message="Found",
            failure_message="Incorrect return value when p == 0",
        )
        naive_annots.append(partial.before(next_partial))
        partial = next_partial

    return B


dac_annots = []

def matpow_dac(A, p, collection=None):
    """
    Raise a matrix to a non-negative power.

    Parameters
    ----------
    A          : list[list[int]]
        the matrix
    p          : int
        the power
    collection : pybryt.Collection
        a collection of annotations to which the intermediate matrices will be added

    Returns
    -------
    list[list[int]]
        the exponentiated matrix
    """
    if p == 0:
        d = diag(1, len(A))
        dac_annots.append(pybryt.Value(
            d,
            name="identity",
            success_message="Returned identity when p == 0",
            failure_message="Incorrect return value when p == 0",
        ))
        return d

    elif p == 1:
        ret = A
        dac_annots.append(pybryt.Value(
            ret,
            name="1st-power",
            success_message="Returned unaltered matrix when p == 1",
            failure_message="Incorrect return value when p == 1",
        ))
        return ret

    should_track = False
    if collection is None:
        collection = pybryt.Collection(
            enforce_order=True,
            success_message="Found the correct sequence of partial powers",
            failure_message="Did not find the correct sequence of partial powers",
        )
        dac_annots.append(collection)
        should_track = True

    P = matpow_dac(A, p // 2, collection=collection)
    collection.add(pybryt.Value(P))

    if p % 2 == 0:
        ret = matmul(P, P)
    else:
        ret = matmul(A, matmul(P, P))

    if should_track:
        dac_annots.append(pybryt.Value(
            ret,
            name="return-value",
            success_message="Returned correct value",
            failure_message="Incorrect return value",
        ))

    return ret
