import numpy as np
from numpy import sparse as sp


def discreteSecondDiff(n):
    return sp.diags([-2, 1, 1], offsets=[0, -1, 1], shape=(n, n))


def vectorizedLaplace(n, m):
    return sp.kron(sp.eye(m), discreteSecondDiff(n)) * sp.kron(
        sp.eye(n), discreteSecondDiff(m)
    )
