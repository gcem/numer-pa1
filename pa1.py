import numpy as np
import scipy.sparse as sp


def discreteSecondDiff(n: int):
    return sp.diags([-2, 1, 1], offsets=[0, -1, 1], shape=(n, n))


def forwardDiff(n: int):
    return sp.diags([1, -1], offsets=[0, 1], shape=(n, n))


def backwardDiff(n: int):
    return sp.diags([1, -1], offsets=[0, -1], shape=(n, n))


def vectorizedLaplace(n: int, m: int):
    return sp.csr_matrix(sp.kron(sp.eye(m), discreteSecondDiff(n)) +
                         sp.kron(discreteSecondDiff(m), sp.eye(n)),
                         dtype=float)


def gradient(data: np.ndarray):
    (n, m) = data.shape
    return np.dstack(
        [forwardDiff(n) @ data, data @ forwardDiff(m).transpose()])


def squaredNorms(field: np.ndarray):
    return np.multiply(field, field).sum(axis=2)


def divergence(field: np.ndarray):
    (n, m, _) = field.shape
    return (backwardDiff(n) @ field[..., 0] +
            field[..., 1] @ backwardDiff(m).transpose())


def setConditionsAtBoundary(originalA: sp.spmatrix,
                            originalb: sp.spmatrix | np.ndarray,
                            target: np.ndarray, x: int, y: int,
                            sourceShape: tuple):
    (n, m) = sourceShape
    # left and right (first and last columns) are easy:
    A = sp.vstack(
        [sp.eye(n, n * m), originalA[n:-n],
         sp.eye(n, n * m, k=n * m - n)],
        format='csr',
        dtype=float)
    b = originalb.copy()
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


def getSystem(source: np.ndarray, target: np.ndarray, y: int, x: int):
    # looking for A and b in
    # Ax=b
    (n, m) = source.shape
    laplace = vectorizedLaplace(n, m)

    A = laplace
    b = laplace @ source.flatten('F').transpose()

    return setConditionsAtBoundary(A, b, target, x, y, source.shape)


def getSystemForCommonFeatures(source: np.ndarray, target: np.ndarray, y: int,
                               x: int):
    (n, m) = source.shape
    A = vectorizedLaplace(n, m)

    sourceGradient = gradient(source)
    targetGradient = gradient(target[y:y + n, x:x + m])

    targetFeatures = squaredNorms(targetGradient) > squaredNorms(
        sourceGradient)
    maxGradient = sourceGradient.copy()
    maxGradient[targetFeatures, ...] = targetGradient[targetFeatures]

    b = divergence(maxGradient).flatten('F').transpose()

    return setConditionsAtBoundary(A, b, target, x, y, source.shape)


def clone(source: np.ndarray,
          target: np.ndarray,
          y: int,
          x: int,
          method: str | None = None):
    sourceF = source.astype(float)
    targetF = target.astype(float)

    getSystemFunction = getSystemForCommonFeatures if method == "common" else getSystem
    (A, b) = getSystemFunction(sourceF, targetF, y, x)
    solution = sp.linalg.spsolve(A, b)
    # solution, info = sp.linalg.cg(A, b, maxiter=5000)
    # solution, istop, iter, *_ = sp.linalg.lsqr(A, b)
    # TODO is info 0?
    result = target.copy()
    (n, m) = source.shape
    result[y:y + n,
           x:x + m] = np.reshape(solution, source.shape,
                                 'F').round().clip(0, 255).astype('uint8')
    return result
