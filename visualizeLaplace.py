import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1


def imshowThreeColors(matrix):
    (n, m) = matrix.shape
    # # alternative:
    # ski.io.imshow(matrix, plugin="matplotlib")

    aspectRatio = m / n

    # prepare figure
    legendWidthRatio = 10
    fig = plt.figure()
    fig.subplots(1, 2, width_ratios=[legendWidthRatio, 1])

    # plot the data
    ax = fig.axes[0]
    ax.set_title(f"Vektorisierter Laplace-Operator ({n}, {m})")
    ax.imshow(matrix, extent=(0, m, n, 0), interpolation=None, cmap='gray')
    ax.minorticks_on()
    ax.grid(True, 'both')

    # plot the legend
    axColor = fig.axes[1]
    dummyData = np.array([1, 0, -4]).repeat(100)
    dummyData = np.vstack((dummyData, dummyData)).transpose()
    axColor.imshow(dummyData, aspect=1 / 10, cmap='gray')
    axColor.set_title("Eintrag der Matrix")
    axColor.set_yticks(np.linspace(0, dummyData.shape[0], 7)[1::2])
    axColor.set_yticklabels(["1", "0", "-4"])
    axColor.xaxis.set_visible(False)

    plt.show()


def visualizeLaplace():
    laplace = pa1.vectorizedLaplace(5, 7).toarray()
    # display the result
    imshowThreeColors(laplace)


if __name__ == "__main__":
    visualizeLaplace()
