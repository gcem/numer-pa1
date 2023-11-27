import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1
import threading


def run():
    bird = ski.io.imread("images/bird.jpg").astype(float)
    plane = ski.io.imread("images/plane.jpg").astype(float)

    y = 24
    x = 350

    # bird = ski.io.imread("images/bird_ds.jpg").astype(float)
    # plane = ski.io.imread("images/plane_ds.jpg").astype(float)

    # y = 2
    # x = 20

    result = np.zeros_like(bird)

    # for rgb in range(3):
    #     result[..., rgb] = pa1.clone(plane[..., rgb], bird[..., rgb], y, x)
    def doChannel(rgb: int):
        result[..., rgb] = pa1.clone(plane[..., rgb], bird[..., rgb], y, x)

    threads = list()
    for rgb in range(3):
        th = threading.Thread(target=doChannel, args=(rgb, ))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    clippedResult = np.clip(result.round().astype('uint8'), 0, 255)
    ski.io.imsave("images/result.jpg", clippedResult)

    ski.io.imsave("images/result_red.jpg", clippedResult[..., 0])
    ski.io.imsave("images/result_green.jpg", clippedResult[..., 1])
    ski.io.imsave("images/result_blue.jpg", clippedResult[..., 2])

    ski.io.imshow(clippedResult, plugin="matplotlib")
    plt.show()


if __name__ == "__main__":
    run()
