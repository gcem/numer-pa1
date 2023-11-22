import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1


def run():
    bird = ski.io.imread("images/bird.jpg")
    plane = ski.io.imread("images/plane.jpg")

    y = 24
    x = 320

    # bird = ski.io.imread("images/bird_ds.jpg")
    # plane = ski.io.imread("images/plane_ds.jpg")

    # y = 2
    # x = 20

    result = np.zeros_like(bird)
    for rgb in range(3):
        result[..., rgb] = pa1.clone(plane[..., rgb], bird[..., rgb], y, x)

    ski.io.imsave("images/result.jpg", result)
    ski.io.imshow(result, plugin="matplotlib")
    plt.show()


run()
