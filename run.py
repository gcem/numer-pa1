import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1


def run():
    bird = ski.io.imread("images/bird.jpg")
    plane = ski.io.imread("images/plane.jpg")

    y = 24
    x = 320

    result = np.zeros_like(bird)
    for rgb in range(3):
        result[..., rgb] = pa1.clone(plane[..., rgb], bird[..., rgb], y, x)

    ski.io.imshow(result)


run()
