import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pa1
import threading
import show


def run():
    bird = ski.io.imread("images/bird.jpg")
    plane = ski.io.imread("images/plane.jpg")

    y = 24
    x = 350

    # bird = ski.io.imread("images/bird_ds.jpg")
    # plane = ski.io.imread("images/plane_ds.jpg")

    # y = 2
    # x = 20

    result1 = np.zeros_like(bird)
    result2 = np.zeros_like(bird)

    # for rgb in range(3):
    #     result[..., rgb] = pa1.clone(plane[..., rgb],
    #                                  bird[..., rgb],
    #                                  y,
    #                                  x,
    #                                  method="keepBoth")
    def doChannel(rgb: int, result: np.ndarray, method: str | None):
        result[..., rgb] = pa1.clone(plane[..., rgb],
                                     bird[..., rgb],
                                     y,
                                     x,
                                     method=method)

    threads = list()
    for rgb in range(3):
        th1 = threading.Thread(target=doChannel, args=(rgb, result1, None))
        threads.append(th1)
        th1.start()
        th2 = threading.Thread(target=doChannel,
                               args=(rgb, result2, 'keepBoth'))
        threads.append(th2)
        th2.start()
    for th in threads:
        th.join()

    # # ski.io.imsave("images/result.jpg", result)
    # ski.io.imsave("images/result_common.jpg", result)

    # ski.io.imsave("images/result_red.jpg", result[..., 0])
    # ski.io.imsave("images/result_green.jpg", result[..., 1])
    # ski.io.imsave("images/result_blue.jpg", result[..., 2])

    # ski.io.imshow(result, plugin="matplotlib")
    show.showAll(bird, result1, result2)
    plt.show()


if __name__ == "__main__":
    run()
