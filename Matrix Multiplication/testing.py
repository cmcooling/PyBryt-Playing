import random
from solutions import diag, matpow_naive, matpow_dac
import numpy as np

def generate_rand_matrix(n, seed, dmin=-100, dmax=100):
    """
    Generate a square matrix populated with random data.

    Parameters
    ----------
    n    : int
        the size of the matrix
    seed : int
        a seed to use for the RNG
    dmin : int
        the minimum value to allow when generating data
    dmax : int
        the maximum value to allow when generating data

    Returns
    -------
    list[list[int]]
        the matrix
    """
    rng = random.Random(seed)
    return [[rng.randrange(dmin, dmax) for _ in range(n)] for _ in range(n)]

def test_matpow(matpow):
    """
    Tests an implementation of matrix exponentiation against NumPy.

    Parameters
    ----------
    matpow : callable[[list[list[int]], int], list[list[int]]]
        the matrix power function

    Raises
    ------
    AssertionError
        if `matpow` returns an incorrect value
    """
    simple_matrix_powers = [
        (diag(0, 5), 0),
        (diag(0, 5), 2),
        (diag(1, 5), 2),
    ]
    for args in simple_matrix_powers:
        mat_pow = matpow(*args)
        assert np.allclose(mat_pow, np.linalg.matrix_power(*args))

    rng = random.Random(42)
    for _ in range(5):
        n, p = rng.randrange(10, 100), rng.randrange(10)
        mat = generate_rand_matrix(n, n, dmin=-10, dmax=10)
        mat_pow = matpow(mat, p)
        assert np.allclose(
            mat_pow, np.linalg.matrix_power(np.array(mat, dtype="int64"), p))

test_matpow(matpow_naive)
test_matpow(matpow_dac)