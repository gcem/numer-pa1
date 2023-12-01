import matplotlib.pyplot as plt
import sys
import run
import numpy as np


def showImage(image: np.ndarray, title: str, location: tuple, fig, cmap=None):
    (h, w, *_) = image.shape

    ax = fig.add_subplot(6, 9, location)
    ax.set_title(title)
    ax.imshow(image, extent=(0, w, h, 0), interpolation=None, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def showColumn(image: np.ndarray, title: str, index: int, fig):
    showImage(image, title, (1 + 3 * index, 21 + 3 * index), fig)
    for rgb in range(3):
        showImage(image[..., rgb], ['red', 'green', 'blue'][rgb],
                  29 + rgb * 9 + 3 * index, fig, 'grey')


def showAll(original: np.ndarray, result1: np.ndarray, result2: np.ndarray):
    # prepare figure
    fig = plt.figure()

    showColumn(original, 'original image', 0, fig)
    showColumn(result1, 'part 1', 1, fig)
    showColumn(result2, 'part 2', 2, fig)


def runAndShow(*runargs, **runkwargs):
    (background, result1, result2) = run.run(*runargs, **runkwargs)
    showAll(background, result1, result2)


if __name__ == "__main__":
    maxThreads = None
    if len(sys.argv) > 1:
        maxThreads = int(sys.argv[-1])

    runAndShow("images/water.jpg", "images/bear.jpg", 50, 10, maxThreads)
    runAndShow("images/bird.jpg", "images/plane.jpg", 25, 380, maxThreads)
    plt.show()
