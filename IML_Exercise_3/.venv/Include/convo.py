import numpy as np
from PIL import Image
import math


def make_kernel(ksize, sigma):
    """
    Create a Gaussian kernel with specified size and sigma.
    """
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    sum_val = 0
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]
    kernel /= sum_val
    return kernel


def slow_convolve(arr, k):
    """
    Apply convolution with the specified kernel and zero padding.
    """
    arr_height, arr_width = arr.shape
    k_height, k_width = k.shape
    pad_height = k_height // 2
    pad_width = k_width // 2
    k_flipped = np.flip(k)
    padded_arr = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    convolved_arr = np.zeros_like(arr)

    for i in range(arr_height):
        for j in range(arr_width):
            region = padded_arr[i:i + k_height, j:j + k_width]
            convolved_arr[i, j] = np.sum(region * k_flipped)

    return convolved_arr


if __name__ == '__main__':
    k = make_kernel(3, 1)
    # TODO: chose the image you prefer
    # im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    im = np.array(Image.open('contrast.jpg').convert('L'))  # convert to grayscale
    blurred = slow_convolve(im, k)
    unsharp_mask = im - blurred
    sharpened = im + unsharp_mask
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(sharpened)
    result_image.save('output.jpg')
    result_image.show()
