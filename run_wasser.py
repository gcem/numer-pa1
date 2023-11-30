import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1
import threading


def run():
    bear = ski.io.imread("images/bear.jpg")
    water = ski.io.imread("images/water.jpg")

    y = 50
    x = 10

    # bird = ski.io.imread("images/bird_ds.jpg")
    # plane = ski.io.imread("images/plane_ds.jpg")

    # y = 2
    # x = 20

    result = np.zeros_like(water)

    # for rgb in range(3):
    #     result[..., rgb] = pa1.clone(plane[..., rgb], bird[..., rgb], y, x)
    def doChannel(rgb: int):
        result[..., rgb] = pa1.clone(bear[..., rgb],
                                     water[..., rgb],
                                     y,
                                     x,
                                     method="common")

    threads = list()
    for rgb in range(3):
        th = threading.Thread(target=doChannel, args=(rgb, ))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    # ski.io.imsave("images/result.jpg", result)
    ski.io.imsave("images/result_waterbear.jpg", result)

    ski.io.imsave("images/result_red.jpg", result[..., 0])
    ski.io.imsave("images/result_green.jpg", result[..., 1])
    ski.io.imsave("images/result_blue.jpg", result[..., 2])

    ski.io.imshow(result, plugin="matplotlib")
    plt.show()


if __name__ == "__main__":
    run()
