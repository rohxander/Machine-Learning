'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    Calculate similarity of Ring features using Euclidean distance.
    :param Rx: Ring features of Person X (numpy array)
    :param Ry: Ring features of Person Y (numpy array)
    :return: Similarity index (float)
    '''
    n = len(Rx)
    r_distance = (1/n)*np.sum(np.abs(Rx - Ry))
    return r_distance


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    Calculate similarity of Fan features using Cosine similarity.
    :param Thetax: Fan features of Person X (numpy array)
    :param Thetay: Fan features of Person Y (numpy array)
    :return: Similarity index (float)
    '''
    mean_tx= np.mean(Thetax)
    mean_ty = np.mean(Thetay)
    lxx = np.sum((Thetax - mean_tx)**2)
    lyy = np.sum((Thetay - mean_ty)**2)
    lxy = np.sum((Thetax - mean_tx)*(Thetay - mean_ty))
    theta_distance = (1 - (lxy ** 2) / (lxx * lyy)) * 100
    return theta_distance
