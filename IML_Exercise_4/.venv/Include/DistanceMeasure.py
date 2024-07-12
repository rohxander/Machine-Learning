'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(R1: np.ndarray, R2: np.ndarray) -> float:
    '''
    Calculate the normalized R distance between two feature vectors.
    :param R1: feature vector 1
    :param R2: feature vector 2
    :return: normalized R distance
    '''
    # Sum of absolute differences
    diff_sum = np.sum(np.abs(R1 - R2))
    # Normalized by the number of features
    norm_diff = diff_sum / len(R1)
    return norm_diff

def calculate_Theta_Distance(T1: np.ndarray, T2: np.ndarray) -> float:
    '''
    Calculate the normalized Theta distance between two feature vectors.
    :param T1: feature vector 1
    :param T2: feature vector 2
    :return: normalized Theta distance
    '''
    # Sum of absolute differences
    diff_sum = np.sum(np.abs(T1 - T2))
    # Normalized by the number of features
    norm_diff = diff_sum / len(T1)
    return norm_diff
