import skimage as ski
import matplotlib.pyplot as plt
import run
import sys


def runAndSave(filenamePrefix: str, *runargs, **runkwargs):
    """Die Argumente sind in `run.run()` dokumentiert.

    Args:
        filenamePrefix (str): Präfix für die Namen der Ausgabedateien
    """
    (_, result1, result2) = run.run(*runargs, **runkwargs)

    ski.io.imsave(filenamePrefix + "_part1.jpg", result1)
    ski.io.imsave(filenamePrefix + "_part2.jpg", result2)


if __name__ == "__main__":
    maxThreads = None
    if len(sys.argv) > 1:
        maxThreads = int(sys.argv[-1])

    runAndSave("output_water", "images/water.jpg", "images/bear.jpg", 50, 10,
               maxThreads)
    runAndSave("output_bird", "images/bird.jpg", "images/plane.jpg", 25, 380,
               maxThreads)
    plt.show()
