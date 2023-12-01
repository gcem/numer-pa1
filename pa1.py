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
    isField2Larger = squaredNorms(field2) > squaredNorms(field1)
    result = field1.copy()
    result[isField2Larger, ...] = field2[isField2Larger, ...]
    return result


def divergence(field: np.ndarray):
    (n, m, _) = field.shape
    return (backwardDiff(n) @ field[..., 0] +
            field[..., 1] @ backwardDiff(m).transpose())


def setConditionsAtBoundary(originalA: sp.spmatrix,
                            originalb: sp.spmatrix | np.ndarray,
                            target: np.ndarray, x: int, y: int,
                            sourceShape: tuple):
    (n, m) = sourceShape

    innerIndices = np.hstack(
        [range(i * n + 1, (i + 1) * n - 1) for i in range(1, m - 1)])
    boundaryIndices = list(set(range(n * m)) - set(innerIndices))

    A = sp.vstack([
        originalA[innerIndices, :],
        sp.eye(n * m, dtype=float, format='csr')[boundaryIndices, :]
    ])
    b = np.hstack([
        originalb[innerIndices],
        target[y:y + n, x:x + m].flatten('F')[boundaryIndices]
    ]) # yapf: disable
    return (A, b)


def getSystem(source: np.ndarray, target: np.ndarray, y: int, x: int):
    (n, m) = source.shape
    laplace = vectorizedLaplace(n, m)

    A = laplace
    b = laplace @ source.flatten('F')

    return setConditionsAtBoundary(A, b, target, x, y, source.shape)


def getSystemForCommonFeatures(source: np.ndarray, target: np.ndarray, y: int,
                               x: int):
    (n, m) = source.shape
    A = vectorizedLaplace(n, m)

    sourceGradient = gradient(source)
    targetGradient = gradient(target[y:y + n, x:x + m])
    maxGradient = vectorWithLargerNorm(sourceGradient, targetGradient)

    b = divergence(maxGradient).flatten('F')

    return setConditionsAtBoundary(A, b, target, x, y, source.shape)


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
