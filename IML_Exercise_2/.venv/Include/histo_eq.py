import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_histogram(img):
    histogram = np.zeros(256)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            histogram[k] += 1
    return histogram


def compute_cdf(histogram):
    cdf = np.zeros(256)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    return cdf / cdf[-1]


def apply_histogram_equalization(image, cdf):
    rows, cols = image.shape
    image_equalized = np.zeros((rows, cols))
    cdf_min = 1
    for i in range(0,256):
        if cdf[i] > 0 and cdf[i] <= cdf_min:
            cdf_min = cdf[i]
    for i in range(rows):
        for j in range(cols):
            image_equalized[i, j] = int((cdf[image[i, j]] - cdf_min) / (1 - cdf_min) * 255)
    return image_equalized


def main():
    image = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)
    histogram = compute_histogram(image)
    cdf = compute_cdf(histogram)
    image_equalized = apply_histogram_equalization(image, cdf)
    cv2.imwrite('kitty.png', image_equalized)

    # print("H", np.sum(histogram[:90]))
    # print("CDF", np.sum(cdf[:90]))


    # plt.imshow(image_equalized, cmap='gray', vmin=0, vmax=255)
    # plt.title('Kitty')
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    main()
