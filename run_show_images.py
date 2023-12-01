import matplotlib.pyplot as plt
import show
import sys
import run


def runAndShow(*runargs, **runkwargs):
    (background, result1, result2) = run.run(*runargs, **runkwargs)
    show.showAll(background, result1, result2)


if __name__ == "__main__":
    maxThreads = None
    if len(sys.argv) > 1:
        maxThreads = int(sys.argv[-1])

    runAndShow("images/water.jpg", "images/bear.jpg", 50, 10, maxThreads)
    runAndShow("images/bird.jpg", "images/plane.jpg", 25, 380, maxThreads)
    plt.show()
