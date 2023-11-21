import numpy as np
import scipy.sparse as sp


def discreteSecondDiff(n: int):
    return sp.diags([-2, 1, 1], offsets=[0, -1, 1], shape=(n, n))


def vectorizedLaplace(n: int, m: int):
    return (sp.kron(sp.eye(m), discreteSecondDiff(n)) +
            sp.kron(sp.eye(n), discreteSecondDiff(m)))


def getSystem(source: np.ndarray, target: np.ndarray, y: int, x: int):
    # looking for A and b in
    # Ax=b

    (n, m) = source.shape  # TODO Check order
    laplace = vectorizedLaplace(n, m)

    A = laplace
    b = laplace @ source.flatten('F').transpose()

    # pixels corresponding to the boundary should be assigned values directly.

    # left and right (first and last columns) are easy:
    A[:n, :] = sp.eye(n, n * m)
    A[(-n):, :] = sp.eye(n, n * m)
    b[:n] = target[y:y + n, x]
    b[(-n):] = target[y:y + n, x + m - 1]

    # top and bottom are cumbersome
    for j in range(1, m - 1):
        iTop = n * j
        iBot = n * (j + 1) - 1
        A[iTop, :] = 0
        A[iBot, :] = 0
        A[iTop, iTop] = 1
        A[iBot, iBot] = 1
        b[iTop] = target[y, x + j]
        b[iBot] = target[y + n - 1, x + j]

    return (A, b)


def clone(source: np.ndarray, target: np.ndarray, y: int, x: int):
    (A, b) = getSystem(source, target, y, x)
    solution, info = sp.linalg.cg(A, b)
    # TODO is info 0?
    result = source.copy()
    (n, m) = result.shape  # TODO Check order
    result[y:y + n, x:x + m] = solution
    return result
