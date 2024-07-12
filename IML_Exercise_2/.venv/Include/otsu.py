import numpy as np

def create_greyscale_histogram(img):
    '''
    Returns a histogram of the given image.
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    histogram = np.zeros(256)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            pixel_value = img[i, j]
            histogram[pixel_value] += 1
    return histogram

def binarize_threshold(img, t):
    '''
    Binarize an image with a given threshold.
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    rows, cols = img.shape
    binarized_img = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if img[i, j] > t:
                binarized_img[i, j] = 255
            else:
                binarized_img[i, j] = 0
    return binarized_img

def p_helper(hist, theta):
    '''
    Compute p0 and p1 using the histogram and the current theta.
    :param hist: histogram
    :param theta: current theta
    :return: p0, p1
    '''
    total = np.sum(hist)
    p0 = 0
    for i in range(theta + 1):
        p0 += hist[i]
    p0 = p0 / total

    p1 = 0
    for i in range(theta + 1, len(hist)):
        p1 += hist[i]
    p1 = p1 / total

    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and mu1 using cumulative sums.
    :param hist: histogram
    :param theta: current theta
    :param p0: probability of the first class
    :param p1: probability of the second class
    :return: mu0, mu1
    '''
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(len(hist)))

    mu0 = cumulative_mean[theta] / p0
    mu1 = (cumulative_mean[-1] - cumulative_mean[theta]) / p1

    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    Calculates theta according to Otsu's method.
    :param hist: 1D array
    :return: threshold (int)
    '''
    total = np.sum(hist)
    current_max, threshold = 0, 0

    for theta in range(256):
        p0, p1 = p_helper(hist, theta)
        if p0 == 0 or p1 == 0:
            continue
        mu0, mu1 = mu_helper(hist, theta, p0, p1)
        between_class_variance = p0 * p1 * (mu0 - mu1) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = theta

    return threshold

def otsu(img):
    '''
    Calculates a binarized image using Otsu's method.
    Hint: reuse the other methods.
    :param img: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    hist = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(hist)
    return binarize_threshold(img, threshold)
