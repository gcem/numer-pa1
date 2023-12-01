import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1
import threading
import show


def run(backgroundFilename: np.ndarray, imageToAddFilename: np.ndarray, y, x):
    background = ski.io.imread(backgroundFilename)
    imageToAdd = ski.io.imread(imageToAddFilename)

    result1 = np.zeros_like(background)
    result2 = np.zeros_like(background)

    def doChannel(rgb: int, result: np.ndarray, method: str | None):
        result[..., rgb] = pa1.clone(imageToAdd[..., rgb],
                                     background[..., rgb],
                                     y,
                                     x,
                                     method=method)

    threads = list()
    for rgb in range(3):
        for result, method in [(result1, None), (result2, 'keepBoth')]:
            th = threading.Thread(target=doChannel, args=(rgb, result, method))
            threads.append(th)
            th.start()
    for th in threads:
        th.join()

    show.showAll(background, result1, result2)


if __name__ == "__main__":
    run("images/water.jpg", "images/bear.jpg", y=50, x=10)
    run("images/bird.jpg", "images/plane.jpg", y=25, x=380)
    plt.show()
