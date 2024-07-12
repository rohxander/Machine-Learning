import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
#
# NO MORE MODULES ALLOWED
#

def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    kernel = np.zeros((ksize, ksize), dtype=float)
    k = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            x = i - k
            y = j - k
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    filtered = convolve(img_in, kernel)

    return kernel, filtered.astype(int)


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gx = convolve(img_in, sobel_x)
    gy = convolve(img_in, sobel_y)

    return gx.astype(int), gy.astype(int)


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    g = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    return g.astype(int), theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle = angle * (180.0 / np.pi) % 180
    if angle < 22.5:
        return 0
    elif angle < 67.5:
        return 45
    elif angle < 112.5:
        return 90
    elif angle < 157.5:
        return 135
    else:
        return 0


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    M, N = g.shape
    Z = np.zeros((M, N))
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = g[i, j + 1]
                    r = g[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = g[i + 1, j - 1]
                    r = g[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = g[i + 1, j]
                    r = g[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = g[i - 1, j - 1]
                    r = g[i + 1, j + 1]

                if (g[i, j] >= q) and (g[i, j] >= r):
                    Z[i, j] = g[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z



def hysteris(max_sup, t_low, t_high):
    """
    Calculate hysteresis thresholding.

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteresis thresholded image (np.ndarray)
    """
    rows, cols = max_sup.shape
    res = np.zeros((rows, cols), dtype=np.uint8)
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(max_sup >= t_high)
    res[strong_i, strong_j] = strong
    weak_i, weak_j = np.where((max_sup >= t_low) & (max_sup < t_high))
    res[weak_i, weak_j] = weak
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if res[i, j] == weak:
                if (res[i + 1, j - 1] == strong or res[i + 1, j] == strong or res[i + 1, j + 1] == strong or
                    res[i, j - 1] == strong or res[i, j + 1] == strong or
                    res[i - 1, j - 1] == strong or res[i - 1, j] == strong or res[i - 1, j + 1] == strong):
                    res[i, j] = strong

    return res


def canny(img):
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

if __name__ == '__main__':

    img = plt.imread('contrast.jpg')
    img = np.mean(img, axis=2)
    canny(img)