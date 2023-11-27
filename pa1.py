import numpy as np
import scipy.sparse as sp


def discreteSecondDiff(n: int):
    return sp.diags([-2, 1, 1], offsets=[0, -1, 1], shape=(n, n))


def vectorizedLaplace(n: int, m: int):
    return sp.csr_matrix(sp.kron(sp.eye(m), discreteSecondDiff(n)) +
                         sp.kron(discreteSecondDiff(m), sp.eye(n)),
                         dtype=float)


def getSystem(source: np.ndarray, target: np.ndarray, y: int, x: int):
    # looking for A and b in
    # Ax=b

    (n, m) = source.shape
    laplace = vectorizedLaplace(n, m)

    A = laplace
    b = laplace @ source.flatten('F').transpose()

    # pixels corresponding to the boundary should be assigned values directly.

    # left and right (first and last columns) are easy:
    A = sp.vstack([sp.eye(n, n * m), A[n:-n],
                   sp.eye(n, n * m, k=n * m - n)],
                  format='csr',
                  dtype=float)
    b[:n] = target[y:y + n, x]
    b[(-n):] = target[y:y + n, x + m - 1]

    # top and bottom are cumbersome
    for j in range(1, m - 1):
        iTop = n * j
        iBot = n * (j + 1) - 1
        A[iTop, :] = sp.eye(1, n * m, k=iTop, format='csr', dtype=float)
        A[iBot, :] = sp.eye(1, n * m, k=iBot, format='csr', dtype=float)
        b[iTop] = target[y, x + j]
        b[iBot] = target[y + n - 1, x + j]

    return (A, b)


def clone(source: np.ndarray, target: np.ndarray, y: int, x: int):
    (A, b) = getSystem(source, target, y, x)
    solution = sp.linalg.spsolve(A, b)
    # solution, info = sp.linalg.cg(A, b, maxiter=5000)
    # solution, istop, iter, *_ = sp.linalg.lsqr(A, b)
    # TODO is info 0?
    result = target.copy()
    (n, m) = source.shape
    result[y:y + n, x:x + m] = np.reshape(solution, source.shape, 'F')
    return result
