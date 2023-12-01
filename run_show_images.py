import matplotlib.pyplot as plt
import sys
import run
import numpy as np


def showImage(image: np.ndarray,
              title: str,
              location: tuple,
              fig,
              cmap: str | None = None):
    """Zeigt das Bild in der Stelle `location` in einem 6 x 9 Grid an.

    Args:
        image (np.ndarray): Das anzuzeigende Bild
        title (str): Titel vom Bild
        location (tuple): Entweder eine Zahl (eine Zelle im Grid) oder Tupel
        (obenLinks, untenRechts). Obere linke Zelle ist 1, obere rechte Zelle 9.
        fig: matplotlib figure
        cmap (str | None, optional): cmap für `imshow` von matplotlib.
    """
    (h, w, *_) = image.shape

    ax = fig.add_subplot(6, 9, location)
    ax.set_title(title)
    ax.imshow(image, extent=(0, w, h, 0), interpolation=None, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def showColumn(image: np.ndarray, title: str, index: int, fig):
    """Zeigt das Bild und seine 3 einfarbigen Komponente in der `index`-ten
    Spalte von einem Grid an.

    Args:
        image (np.ndarray): Das anzuzeigende Bild
        title (str): Titel vom Bild
        index (int): Spalte (0 bis 2)
        fig: matplotlib figure
    """
    showImage(image, title, (1 + 3 * index, 21 + 3 * index), fig)
    for rgb in range(3):
        showImage(image[..., rgb], ["red", "green", "blue"][rgb],
                  29 + rgb * 9 + 3 * index, fig, "grey")


def showAll(original: np.ndarray, result1: np.ndarray, result2: np.ndarray):
    """Zeigt die Bilder in einem Fenster an.

    Args:
        original (np.ndarray): Originalbild
        result1 (np.ndarray): 1. Aufgabe
        result2 (np.ndarray): 2. Aufgabe
    """
    # prepare figure
    fig = plt.figure()

    showColumn(original, "Originalbild", 0, fig)
    showColumn(result1, "1. Teilaufgabe", 1, fig)
    showColumn(result2, "2. Teilaufgabe", 2, fig)


def runAndShow(*runargs, **runkwargs):
    """Die Argumente sind in `run.run()` dokumentiert.
    """
    (background, result1, result2) = run.run(*runargs, **runkwargs)
    showAll(background, result1, result2)


if __name__ == "__main__":
    maxThreads = None
    if len(sys.argv) > 1:
        maxThreads = int(sys.argv[-1])

    runAndShow("images/water.jpg", "images/bear.jpg", 50, 10, maxThreads)
    runAndShow("images/bird.jpg", "images/plane.jpg", 25, 380, maxThreads)
    plt.show()
