'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    _, binarized = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    smoothed = cv2.GaussianBlur(binarized, (5, 5), 0)
    return smoothed


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, [largest_contour], -1, 255, 2)
    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    intersections = []
    for y in range(contour_img.shape[0]):
        if contour_img[y, x] == 255:
            intersections.append(y)
    if len(intersections) >= 6:
        return np.array(intersections[:6])
    return np.array(intersections)



def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    k1_y, k1_x = y1, x1
    k2_y, k2_x = y2, x2
    k3_y = (k1_y + k2_y) // 2
    k3_x = (k1_x + k2_x) // 2
    return k3_y, k3_x


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    src_pts = np.array([k1, k2, k3], dtype="float32")
    # The destination points should form a right-angle triangle
    dst_pts = np.array([k1, [k1[0], k1[1] + 100], [k1[0] + 100, k1[1]]], dtype="float32")
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    return matrix

def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''
    preprocessed = binarizeAndSmooth(img)
    contour_img = drawLargestContour(preprocessed)
    col1, col2 = 10, img.shape[1] - 10
    intersections1 = getFingerContourIntersections(contour_img, col1)
    intersections2 = getFingerContourIntersections(contour_img, col2)
    middle_points = [((y1 + y2) // 2, (x1 + x2) // 2) for y1, y2, x1, x2 in
                     zip(intersections1, intersections2, [col1] * 6, [col2] * 6)]
    k1, k2, k3 = middle_points[0], middle_points[1], middle_points[2]
    transform_matrix = getCoordinateTransform(k1, k2, k3)
    aligned_img = cv2.warpAffine(img, transform_matrix, (img.shape[1], img.shape[0]))
    return aligned_img