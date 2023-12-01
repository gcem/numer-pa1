import numpy as np
import scipy.sparse as sp


def discreteSecondDiff(n: int) -> sp.spmatrix:
    """Erzeugt die n x n Matrix für diskrete zweite Ableitung.

    Args:
        n (int): Dimension

    Returns:
        sp.spmatrix: n x n dünnbesetzte Matrix
    """
    return sp.diags([-2, 1, 1], offsets=[0, -1, 1], shape=(n, n))


def forwardDiff(n: int) -> sp.spmatrix:
    """Erzeugt die n x n Matrix für diskrete Ableitung (vorwärts).

    Args:
        n (int): Dimension

    Returns:
        sp.spmatrix: n x n dünnbesetzte Matrix
    """
    return sp.diags([-1, 1], offsets=[0, 1], shape=(n, n))


def backwardDiff(n: int) -> sp.spmatrix:
    """Erzeugt die n x n Matrix für diskrete Ableitung (rückwärts).

    Args:
        n (int): Dimension

    Returns:
        sp.spmatrix: n x n dünnbesetzte Matrix
    """
    return sp.diags([1, -1], offsets=[0, -1], shape=(n, n))


def vectorizedLaplace(n: int, m: int) -> sp.csr_matrix:
    """Erzeugt den diskreten Laplace-Operator für ein n x m System.

    Args:
        n (int): Dimension
        m (int): Dimension

    Returns:
        sp.csr_matrix: nm x nm dünnbesetzte Matrix
    """
    return sp.csr_matrix(sp.kron(sp.eye(m), discreteSecondDiff(n)) +
                         sp.kron(discreteSecondDiff(m), sp.eye(n)),
                         dtype=float)


def gradient(data: np.ndarray) -> np.ndarray:
    """Findet den (diskreten) Gradienten vom Skalarfeld `data`.

    Args:
        data (np.ndarray): n x m Matrix

    Returns:
        np.ndarray: n x m x 2 Matrix, wobei die 2-Vektoren in der letzten Achse
        die Gradientenvektoren sind
    """
    (n, m) = data.shape
    return np.dstack(
        [forwardDiff(n) @ data, data @ forwardDiff(m).transpose()])


def squaredNorms(field: np.ndarray) -> np.ndarray:
    """Findet die quadrierte 2-Norm vom Vektorfeld `field`.

    Args:
        field (np.ndarray): n x m x dim Matrix, bestehend aus den Vektoren im
        dim-Dimensionalen Vektorraum.

    Returns:
        np.ndarray: n x m Matrix
    """
    return np.multiply(field, field).sum(axis=2)


def vectorWithLargerNorm(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    """Wählt in jedem Punkt den Vektor mit größerer Norm aus den Vektorräumen
    `field1` und `field2`.

    Args:
        field1 (np.ndarray): n x m x dim Matrix
        field2 (np.ndarray): n x m x dim Matrix
    Returns:
        np.ndarray: n x m x dim Matrix
    """
    isField2Larger = squaredNorms(field2) > squaredNorms(field1)
    result = field1.copy()
    result[isField2Larger, ...] = field2[isField2Larger, ...]
    return result


def divergence(field: np.ndarray) -> np.ndarray:
    """Findet die (diskrete) Divergenz vom 2-dimensionalen Vektorfeld `field`.

    Args:
        field (np.ndarray): n x m x 2 Matrix

    Returns:
        np.ndarray: n x  m Matrix
    """
    (n, m, _) = field.shape
    return (backwardDiff(n) @ field[..., 0] +
            field[..., 1] @ backwardDiff(m).transpose())


def setConditionsAtBoundary(originalA: sp.spmatrix,
                            originalb: sp.spmatrix | np.ndarray,
                            target: np.ndarray, x: int, y: int,
                            sourceShape: tuple):
    """Löscht die Zeilen im System Ax = b, die den Randpixels entsprechen. Fügt
    dann neue Zeilen hinzu, die den Wert von x am Rand gleich zum Wert von
    `target` am Rand setzen.

    Args:
        originalA (sp.spmatrix): A in Ax=b
        originalb (sp.spmatrix | np.ndarray): b in Ax=b
        target (np.ndarray): Das Bild, aus dem die Randwerte genommen werden
        sollen
        x (int): x-Koordinate vom oberen linken Pixel in `target` vom Rand
        y (int): y-Koordinate vom oberen linken Pixel in `target` vom Rand
        sourceShape (tuple): 2-Tupel mit (Höhe, Breite)

    Returns:
        _type_: _description_
    """
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
    """Findet ein System Ax=b, dessen Lösung das Bild `source` im Bild `target`
    in der Stelle `y`,`x` (obere linke Ecke) einbettet.

    Nutzt nur den Gradienten vom `source`, deswegen verschwinden die Details vom
    `target` im Zielbereich.

    Args:
        source (np.ndarray): n x m Matrix, das einzubettende Bild
        target (np.ndarray): eine größere Matrix als `source` 
        y (int): y-Koordinate vom oberen linken Pixel in `target` vom
        Zielbereich
        x (int): x-Koordinate vom oberen linken Pixel in `target` vom
        Zielbereich

    Returns:
        Tuple: (A, b)
    """
    (n, m) = source.shape
    laplace = vectorizedLaplace(n, m)

    A = laplace
    b = laplace @ source.flatten('F')

    return setConditionsAtBoundary(A, b, target, x, y, source.shape)


def getSystemKeepBoth(source: np.ndarray, target: np.ndarray, y: int, x: int):
    """Findet ein System Ax=b, dessen Lösung das Bild `source` im Bild `target`
    in der Stelle `y`,`x` (obere linke Ecke) einbettet.

    Nutzt den größeren Gradienten von `source` und `target`, deswegen werden die
    Details aus den beiden Bildern gehalten.

    Args:
        source (np.ndarray): n x m Matrix, das einzubettende Bild
        target (np.ndarray): eine größere Matrix als `source` 
        y (int): y-Koordinate vom oberen linken Pixel in `target` vom
        Zielbereich
        x (int): x-Koordinate vom oberen linken Pixel in `target` vom
        Zielbereich

    Returns:
        Tuple: (A, b)
    """
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
          method: str | None = None) -> np.ndarray:
    """Bettet das Bild `source` ins Bild `target` in der Stelle `y`,`x` (obere 
    linke Ecke) ein.

    Der Zielbereich muss gänzlich im Zielbild (`target`) enthalten sein (diese
    Funktion schneidet die Bilder nicht).

    Args:
        source (np.ndarray): n x m Matrix, das einzubettende Bild
        target (np.ndarray): eine größere Matrix als `source`, das
        Hintergrundbild
        y (int): y-Koordinate vom oberen linken Pixel in `target` vom
        Zielbereich
        x (int): x-Koordinate vom oberen linken Pixel in `target` vom
        Zielbereich
        method (str | None, optional): "keepBoth" oder None. None löst die erste
        Teilaufgabe, "keepBoth" die Zweite. Default ist None

    Returns:
        np.ndarray: Das neue Bild
    """
    (n, m) = source.shape
    sourceF = source.astype(float)
    targetF = target.astype(float)

    getSystemFunction = getSystemKeepBoth if method == "keepBoth" else getSystem
    (A, b) = getSystemFunction(sourceF, targetF, y, x)
    solution = sp.linalg.spsolve(A, b)
    result = target.copy()
    result[y:y + n,
           x:x + m] = np.reshape(solution, source.shape,
                                 'F').round().clip(0, 255).astype('uint8')
    return result
