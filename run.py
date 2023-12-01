import numpy as np
import skimage as ski
import pa1
import threading


def run(backgroundFilename: str, imageToAddFilename: str, y: int, x: int,
        maxThreads: int | None):
    """Löst die beiden Aufgaben. Fügt `imageToAdd` ins `background` in der
    Stelle `y`, `x` ein.

    Args:
        backgroundFilename (str): Pfad
        imageToAddFilename (str): Pfad
        y (int): y-Koordinate
        x (int): x-Koordinate
        maxThreads (int | None): None heißt die maximale Anzahl, zurzeit 6.

    Returns:
        Tuple: (Hintergrund, 1. Aufgabe, 2. Aufgabe)
    """
    background = ski.io.imread(backgroundFilename)
    imageToAdd = ski.io.imread(imageToAddFilename)

    result1 = np.zeros_like(background)
    result2 = np.zeros_like(background)

    def doChannel(rgb: int, result: np.ndarray, method: str | None, semaphore):
        if semaphore:
            semaphore.acquire()
        result[..., rgb] = pa1.clone(imageToAdd[..., rgb],
                                     background[..., rgb],
                                     y,
                                     x,
                                     method=method)
        if semaphore:
            semaphore.release()

    # begrenzt ggf. die Anzahl von threads
    semaphore = threading.BoundedSemaphore(maxThreads) if maxThreads else None
    threads = list()
    for rgb in range(3):
        for result, method in [(result1, None), (result2, 'keepBoth')]:
            th = threading.Thread(target=doChannel,
                                  args=(rgb, result, method, semaphore))
            threads.append(th)
            th.start()
    for th in threads:
        th.join()

    return (background, result1, result2)
