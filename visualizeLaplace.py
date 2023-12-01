import numpy as np
import matplotlib.pyplot as plt
import pa1


def visualizeLaplace():
    n, m = 5, 7
    laplace = pa1.vectorizedLaplace(n, m).toarray()

    # # alternative:
    # ski.io.imshow(laplace, plugin="matplotlib")

    w = h = n * m
    # prepare figure
    legendWidthRatio = 10
    fig = plt.figure()
    fig.subplots(1, 2, width_ratios=[legendWidthRatio, 1])

    # plot the data
    ax = fig.axes[0]
    ax.set_title(f"Vektorisierter Laplace-Operator ({n}, {m})")
    ax.imshow(laplace, extent=(0, h, w, 0), interpolation=None, cmap='grey')
    ax.minorticks_on()
    ax.grid(True, 'both')

    # plot the legend
    axColor = fig.axes[1]
    dummyData = np.array([1, 0, -4]).repeat(100)
    dummyData = np.vstack((dummyData, dummyData)).transpose()
    axColor.imshow(dummyData, aspect=1 / 10, cmap='grey')
    axColor.set_title("Eintrag der Matrix")
    axColor.set_yticks(np.linspace(0, dummyData.shape[0], 7)[1::2])
    axColor.set_yticklabels(["1", "0", "-4"])
    axColor.xaxis.set_visible(False)

    plt.show()


if __name__ == "__main__":
    visualizeLaplace()
