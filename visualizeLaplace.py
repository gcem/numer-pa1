import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1


def imshowThreeColors(matrix):
    (n, m) = matrix.shape
    ski.io.imshow(np.stack([(matrix * 255).round().clip(0, 255),
                            np.zeros_like(matrix),
                            np.zeros_like(matrix)],
                           axis=2).astype('uint8'),
                  cmap='grey',
                  plugin="matplotlib")
    # aspectRatio = m / n

    # # prepare figure
    # height = 400  # px
    # legendWidthRatio = 10
    # fig = plt.figure()
    # fig.subplots(1, 2, width_ratios=[legendWidthRatio, 1])

    # # plot data
    # ax = fig.axes[0]
    # ax.set_title(f"Vektorisierter Laplace-Operator ({n}, {m})")
    # ax.imshow(matrix, extent=(0, m, n, 0), interpolation=None, cmap='gray')

    # # plot the legend
    # axColor = fig.axes[1]
    # gradient = np.linspace(1, 0, 3).repeat(100)
    # gradient = np.vstack((gradient, gradient)).transpose()
    # axColor.imshow(gradient, aspect=1 / 10, cmap='gray')
    # axColor.set_title("Eintrag der Matrix")
    # axColor.set_yticks(np.linspace(0, gradient.shape[0], 7)[1::2])
    # axColor.set_yticklabels(["-4", "1", "0"])
    # axColor.xaxis.set_visible(False)

    plt.show()


def visualizeLaplace():
    laplace = pa1.vectorizedLaplace(5, 7).toarray()
    # display the result
    imshowThreeColors(abs(laplace / 4.))


if __name__ == "__main__":
    visualizeLaplace()
