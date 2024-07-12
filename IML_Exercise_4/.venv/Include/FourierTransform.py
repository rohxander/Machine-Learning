'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!



def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to cartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    y = int(shape[0] / 2 + r * np.sin(theta))
    x = int(shape[1] / 2 + r * np.cos(theta))
    return y, x

def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    Return Magnitude in Decibel
    :param img:
    :return:
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring --> theta/sampling rate
    :return: feature vector of k features
    '''
    center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
    radii = np.linspace(0, min(center), k + 1)[1:]
    features = np.zeros(k)

    for i, r in enumerate(radii):
        samples = []
        for theta in np.linspace(0, 2 * np.pi, sampling_steps, endpoint=False):
            y, x = polarToKart(magnitude_spectrum.shape, r, theta)
            samples.append(magnitude_spectrum[y, x])
        features[i] = np.mean(samples)

    return features

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
    center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
    radius = min(center)
    angles = np.linspace(0, 2 * np.pi, k + 1)[:-1]
    features = np.zeros(k)

    for i, theta in enumerate(angles):
        samples = []
        for r in np.linspace(0, radius, sampling_steps):
            y, x = polarToKart(magnitude_spectrum.shape, r, theta)
            samples.append(magnitude_spectrum[y, x])
        features[i] = np.mean(samples)

    return features

def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    return R, T