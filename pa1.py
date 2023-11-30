import numpy as np
import scipy.sparse as sp


def discreteSecondDiff(n: int):
    return sp.diags([-2, 1, 1], offsets=[0, -1, 1], shape=(n, n))


def forwardDiff(n: int):
    return sp.diags([-1, 1], offsets=[0, 1], shape=(n, n))


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


def vectorWithLargerNorm(field1: np.ndarray, field2: np.ndarray):
    takeField2 = squaredNorms(field2) > squaredNorms(field1)
    result = field1.copy()
    result[takeField2, ...] = field2[takeField2, ...]
    return result


def divergence(field: np.ndarray):
    (n, m, _) = field.shape
    return (backwardDiff(n) @ field[..., 0] +
            field[..., 1] @ backwardDiff(m).transpose())


def setConditionsAtBoundary(A: sp.spmatrix, b: sp.spmatrix | np.ndarray,
                            target: np.ndarray, x: int, y: int,
                            sourceShape: tuple):
    (n, m) = sourceShape
    boundaryIndices = np.zeros(2 * (n + m) - 4, dtype=int)
    boundaryIndices[:n] = range(n)
    boundaryIndices[-n:] = range(n * (m - 1), n * m)
    topIndices = np.arange(n, n * (m - 1), step=n)
    boundaryIndices[n:-n:2] = topIndices
    boundaryIndices[n + 1:-n:2] = topIndices + (n - 1)

    A[boundaryIndices, :] = sp.eye(n * m, format='csr')[boundaryIndices, :]
    b[boundaryIndices] = target[y:y + n, x:x + m].flatten('F')[boundaryIndices]


def getSystem(source: np.ndarray, target: np.ndarray, y: int, x: int):
    (n, m) = source.shape
    laplace = vectorizedLaplace(n, m)

    A = laplace
    b = laplace @ source.flatten('F').transpose()

    setConditionsAtBoundary(A, b, target, x, y, source.shape)
    return (A, b)


def getSystemForCommonFeatures(source: np.ndarray, target: np.ndarray, y: int,
                               x: int):
    (n, m) = source.shape
    A = vectorizedLaplace(n, m)

    sourceGradient = gradient(source)
    targetGradient = gradient(target[y:y + n, x:x + m])
    maxGradient = vectorWithLargerNorm(sourceGradient, targetGradient)

    b = divergence(maxGradient).flatten('F').transpose()

    setConditionsAtBoundary(A, b, target, x, y, source.shape)
    return (A, b)


def clone(source: np.ndarray,
          target: np.ndarray,
          y: int,
          x: int,
          method: str | None = None):
    (n, m) = source.shape
    sourceF = source.astype(float)
    targetF = target.astype(float)

    getSystemFunction = getSystemForCommonFeatures if method == "common" else getSystem
    (A, b) = getSystemFunction(sourceF, targetF, y, x)
    solution = sp.linalg.spsolve(A, b)
    result = target.copy()
    result[y:y + n,
           x:x + m] = np.reshape(solution, source.shape,
                                 'F').round().clip(0, 255).astype('uint8')
    return result
