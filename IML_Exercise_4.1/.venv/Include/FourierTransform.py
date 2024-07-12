'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!
def crop_to_opposite_aspect(image, ref_point):
    '''
    Crop the image to the opposite aspect ratio with the reference point at the center of the left edge of the new image.
    :param image: Input image (grayscale or color).
    :param ref_point: Tuple (y, x) representing the reference point (center of the left edge of the new image).
    :return: Cropped image.
    '''
    height, width = image.shape[:2]

    # Calculate the new dimensions based on the opposite aspect ratio
    if height > width:
        new_width = height
        new_height = width
    else:
        new_width = height
        new_height = width

    y, x = ref_point

    # Calculate the top-left corner of the new cropping region
    top_left_y = max(0, y - new_height // 2)
    top_left_x = x

    # Ensure the crop is within image boundaries
    if top_left_x + new_width > width:
        top_left_x = width - new_width

    if top_left_y + new_height > height:
        top_left_y = height - new_height

    # Crop the image
    cropped_image = image[top_left_y:top_left_y + new_height, top_left_x:top_left_x + new_width]

    return cropped_image

def crop_square(image,size):
    '''
    Crop a square sub-image centered at a given point.
    :param image: Input image (grayscale or color).
    :param center: Tuple of (y, x) indicating the center of the square.
    :param size: Size of the square sub-image.
    :return: Cropped square sub-image.
    '''
    x , y = (image.shape[0] // 2, image.shape[1] // 2)
    half_size = size // 2
    top_left = (max(0, y - half_size), max(0, x - half_size))
    bottom_right = (min(image.shape[0], y + half_size), min(image.shape[1], x + half_size))
    cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return cropped_image

def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    height, width = shape
    center_x = width // 2
    center_y = height // 2
    x_component = r * np.cos(theta)
    y_component = r * np.sin(theta)
    x = center_x + x_component
    y = center_y + y_component
    return y, x

def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    Return Magnitude in Decibel
    :param img:
    :return:
    '''
    ft = np.fft.fft2(img)
    ft_shifted = np.fft.fftshift(ft)
    magnitude_spectrum = np.abs(ft_shifted)
    magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum + 1e-8)
    return magnitude_spectrum_db


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring --> theta/sampling rate
    :return: feature vector of k features
    '''
    samples = np.zeros(k)
    h, w = magnitude_spectrum.shape
    for i in range(1, k + 1):
        theta = 0
        while theta <= np.pi:
            for r in range(k * (i - 1), k * i + 1):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                if 0 <= x < w and 0 <= y < h:
                    samples[i - 1] += magnitude_spectrum[int(y), int(x)]
            theta += np.pi / (sampling_steps - 1)
    return samples


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area --> theta/sampling rate
    :return: feature vector of length k
    """
    samples = np.zeros(k)

    for i in range(k):
        theta_start = i * np.pi / k
        theta_end = (i + 1) * np.pi / k
        theta_values = np.linspace(theta_start, theta_end, sampling_steps)

        for theta in theta_values:
            for r in range(0, 36):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                y = int(np.clip(y, 0, magnitude_spectrum.shape[0] - 1))
                x = int(np.clip(x, 0, magnitude_spectrum.shape[1] - 1))
                samples[i] += magnitude_spectrum[y, x]

    return samples



def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    # img = crop_square(img, min(img.shape)-40)
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)

    return R, T
