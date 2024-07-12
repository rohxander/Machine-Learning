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
def crop_square(image, ref_point, square_length):
    '''
    Crop a square portion of the image with the reference point at the center of the left edge of the new square image.
    :param image: Input image (grayscale or color).
    :param ref_point: Tuple (y, x) representing the reference point (center of the left edge of the new image).
    :param square_length: Length of the sides of the square to be cropped.
    :return: Cropped square image.
    '''
    height, width = image.shape[:2]
    y, x = ref_point

    # Calculate the top-left corner of the cropping region
    top_left_y = max(0, y + square_length // 3)
    top_left_x = x


    # Crop the image
    cropped_image = image[top_left_y:top_left_y + square_length  , top_left_x:top_left_x + square_length ]

    return cropped_image


def get_center(M):
    alpha_1 = 1 - M[0, 0]
    beta = M[0, 1]
    # cv2 formal naming: x = first center point coordinate
    cx = np.round(((M[1, 2] * alpha_1) - (beta * M[0, 2])) / (np.power(beta, 2) + np.power(alpha_1, 2)))
    cy = np.round((M[0, 2] + (beta * cx)) / alpha_1)
    return cx, cy

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
    binary_img = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)[1]
    return cv2.GaussianBlur(binary_img, (5, 5), 0)

def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_img = np.zeros_like(img)
        cv2.drawContours(contour_img,[largest_contour], -1,(255, 255, 255), 2)
    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    ys = []
    contour_flag = False
    for y in range(5, contour_img.shape[0]):
        if contour_img[y,x] == 255:
            contour_flag = True
        elif contour_img[y,x] == 0 and contour_flag == True:
            ys.append(y-2)
            contour_flag = False
    return np.array(ys[:6])


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
    slope = (y2 - y1) / (x2 - x1)
    const = y1 - (slope * x1)
    # print("slope=",slope,"constant=",const)
    for x in range(img.shape[1]):
        # print(img[int((slope*x)+const),x])
        if img[int((slope*x)+const),x] == 255:
            ky=x
            kx=int((slope*x)+const)
            return (kx, ky)

def getCoordinateTransform(k1, k2, k3) -> tuple:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    slope = (k3[0] - k1[0]) / (k3[1] - k1[1])
    const = k3[0] - slope * k3[1]
    perpendicular_slope = -1 / slope
    c1 = k2[0] - perpendicular_slope * k2[1]

    x_intersect = (c1 - const) / (slope - perpendicular_slope)
    y_intersect = slope * x_intersect + const

    angle = np.arctan(perpendicular_slope)
    rotation_matrix = cv2.getRotationMatrix2D((y_intersect, x_intersect), np.degrees(angle),scale = 1.0)
    return rotation_matrix

def get_center(M):
    alpha_1 = 1 - M[0, 0]
    beta = M[0, 1]
    # cv2 formal naming: x = first center point coordinate
    cx = np.round(((M[1, 2] * alpha_1) - (beta * M[0, 2])) / (np.power(beta, 2) + np.power(alpha_1, 2)))
    cy = np.round((M[0, 2] + (beta * cx)) / alpha_1)
    return cx, cy

def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    preprocessed_img = binarizeAndSmooth(img)
    contour_img = drawLargestContour(preprocessed_img)
    x1, x2 = 10,25
    y_values1 = getFingerContourIntersections(contour_img, x1)
    y_values2 = getFingerContourIntersections(contour_img, x2)

    y11 = (y_values1[0] + y_values1[1]) / 2
    y12 = (y_values1[2] + y_values1[3]) / 2
    y13 = (y_values1[4] + y_values1[5]) / 2

    y21 = (y_values2[0] + y_values2[1]) / 2
    y22 = (y_values2[2] + y_values2[3]) / 2
    y23 = (y_values2[4] + y_values2[5]) / 2

    k1 = findKPoints(contour_img, y11, x1, y21, x2)
    k2 = findKPoints(contour_img, y12, x1, y22, x2)
    k3 = findKPoints(contour_img, y13, x1, y23, x2)


    cv2.line(contour_img, (k1[1], k1[0]), (k3[1], k3[0]), (255, 0, 0), 2)
    C = getCoordinateTransform(k1, k2, k3)
    cx, cy = get_center(C)
    transformed_img = cv2.warpAffine(img, C, (img.shape[1], img.shape[0]))
    cv2.circle(transformed_img, (int(cx), int(cy)), 5, (0, 0, 0), thickness=-1)
    # cv2.line(transformed_img, (cx.astype(int), cy.astype(int)), (k1[1], k1[0]), (255, 0, 0), 2)
    # cv2.line(transformed_img, (cx.astype(int), cy.astype(int)), (k2[1], k2[0]), (255, 0, 0), 2)
    # cv2.line(transformed_img, (cx.astype(int), cy.astype(int)), (k3[1], k3[0]), (255, 0, 0), 2)
    transformed_img = crop_square(transformed_img,(int(cx), int(cy)),100)
    return transformed_img


