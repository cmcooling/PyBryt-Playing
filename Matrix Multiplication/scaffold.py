import hashlib
from solutions import diag
import random
from testing import generate_rand_matrix
import json
from solutions import matmul
import pybryt
from reference_implementations import naive_ref, dac_ref
from solutions import matpow_dac

hash_matrix = lambda m: hashlib.sha256(str(m).encode()).hexdigest()

def matpow(A, p):
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

    P = matpow(A, p // 2)
    if p % 2 == 0:
        return matmul(P, P)
    else:
        return matmul(A, matmul(P, P))

hashes = []

simple_matrix_powers = [
    (diag(0, 5), 0),
    (diag(0, 5), 2),
    (diag(1, 5), 2),
]
for args in simple_matrix_powers:
    mat_pow = matpow(*args)
    hashes.append(hash_matrix(mat_pow))

rng = random.Random(42)
for _ in range(5):
    n, p = rng.randrange(10, 100), rng.randrange(10)
    mat = generate_rand_matrix(n, n, dmin=-10, dmax=10)
    mat_pow = matpow(mat, p)
    hashes.append(hash_matrix(mat_pow))

with open("hashes.json", "w+") as f:
    json.dump(hashes, f)





def test_matpow(matpow):
    """
    Tests an implementation of matrix exponentiation.

    Parameters
    ----------
    matpow : callable[[list[list[int]], int], list[list[int]]]
        the matrix power function

    Raises
    ------
    AssertionError
        if `matpow` returns an incorrect value
    """
    with open("hashes.json") as f:
        hashes = json.load(f)

    hashes = iter(hashes)
    simple_matrix_powers = [
        (diag(0, 5), 0),
        (diag(0, 5), 2),
        (diag(1, 5), 2),
    ]
    for args in simple_matrix_powers:
        mat_pow = matpow(*args)
        assert hash_matrix(mat_pow) == next(hashes)

    rng = random.Random(42)
    for _ in range(5):
        n, p = rng.randrange(10, 100), rng.randrange(10)
        mat = generate_rand_matrix(n, n, dmin=-10, dmax=10)
        mat_pow = matpow(mat, p)
        assert hash_matrix(mat_pow) == next(hashes)

with pybryt.check([naive_ref, dac_ref]):
    test_matpow(matpow_dac)