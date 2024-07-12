'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel, Charly, Max
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

from FourierTransform import calcuateFourierParameters
from FourierTransform import crop_square
from FourierTransform import calculateMagnitudeSpectrum
from DistanceMeasure import calculate_Theta_Distance, calculate_R_Distance
from PalmprintAlignmentAutomatic import palmPrintAlignment

k = 8
samplingSize = 200

img1 = cv2.imread('Hand1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Hand2.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('Hand3.jpg', cv2.IMREAD_GRAYSCALE)

img1_aligned = palmPrintAlignment(img1)
img2_aligned = palmPrintAlignment(img2)
img3_aligned = palmPrintAlignment(img3)

###########################

crop_img_1 = crop_square(img1_aligned, int(min(img1_aligned.shape)/2.2))
crop_img_2 = crop_square(img2_aligned, int(min(img2_aligned.shape)/2.2))
crop_img_3 = crop_square(img3_aligned, int(min(img3_aligned.shape)/2.2))

magnitude_spectrum_1 = calculateMagnitudeSpectrum(img1_aligned)
magnitude_spectrum_2 = calculateMagnitudeSpectrum(img2_aligned)
magnitude_spectrum_3 = calculateMagnitudeSpectrum(img3_aligned)

###########################
RX, ThetaX = calcuateFourierParameters(img1_aligned, k, samplingSize)
RY, ThetaY = calcuateFourierParameters(img2_aligned, k, samplingSize)
RZ, ThetaZ = calcuateFourierParameters(img3_aligned, k, samplingSize)

DR_xx = calculate_R_Distance(RX, RX)
DTheta_xx = calculate_Theta_Distance(ThetaX, ThetaX)

DR_xy = calculate_R_Distance(RX, RY)
DTheta_xy = calculate_Theta_Distance(ThetaX, ThetaY)

DR_yx = calculate_R_Distance(RY, RX)
DTheta_yx = calculate_Theta_Distance(ThetaY, ThetaX)

DR_xz = calculate_R_Distance(RX, RZ)
DTheta_xz = calculate_Theta_Distance(ThetaX, ThetaZ)

DR_yz = calculate_R_Distance(RY, RZ)
DTheta_yz = calculate_Theta_Distance(ThetaY, ThetaZ)

print("DR: Ring-like area")
print("DTheta: Fan-like area")
print("The smaller the value, the better the match. d=0: completely identical")

print("1-1")
print("DR: " + format(np.round(DR_xx, 1) / samplingSize))
print("DTheta :" + format(np.round(DTheta_xx, 1)))

print("2-1")
print("DR: " + format(np.round(DR_yx / samplingSize, 1)))
print("DTheta :" + format(np.round(DTheta_yx, 1)))

print("1-2")
print("DR: " + format(np.round(DR_xy / samplingSize, 1)))
print("DTheta :" + format(np.round(DTheta_xy, 1)))

print("1-3")
print("DR: " + format(np.round(DR_xz / samplingSize, 1)))
print("DTheta :" + format(np.round(DTheta_xz, 1)))

print("2-3")
print("DR: " + format(np.round(DR_yz / samplingSize, 1)))
print("DTheta :" + format(np.round(DTheta_yz, 1)))

# (ax1, ax2, ax3),
# plotting comparison images of preprocessing
FinalFig, ((ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(3, 3)
# ax1.imshow(img1, 'gray')
# ax1.axis('off')
# # ax1.set_title('PP 1 unaligned')
#
# ax2.imshow(img2, 'gray')
# ax2.axis('off')
# # ax2.set_title('PP 2 unaligned')
#
# ax3.imshow(img3, 'gray')
# ax3.axis('off')
# # ax3.set_title('PP 3 unaligned')

ax4.imshow(img1_aligned, 'gray')
ax4.axis('off')
# ax4.set_title('PP 1 aligned')

ax5.imshow(img2_aligned, 'gray')
ax5.axis('off')
# ax5.set_title('PP 2 aligned')

ax6.imshow(img3_aligned, 'gray')
ax6.axis('off')
# ax6.set_title('PP 3 aligned')

ax7.imshow(crop_img_1, 'gray')
ax7.axis('off')
# ax7.set_title('Crop 1')

ax8.imshow(crop_img_2, 'gray')
ax8.axis('off')
# ax8.set_title('Crop 2')

ax9.imshow(crop_img_3, 'gray')
ax9.axis('off')
# ax9.set_title('Crop 3')

ax10.imshow(magnitude_spectrum_1, 'gray')
ax10.axis('off')
# ax10.set_title('Mag Spec 1')

ax11.imshow(magnitude_spectrum_2, 'gray')
ax11.axis('off')
# ax11.set_title('Mag Spec 2')

ax12.imshow(magnitude_spectrum_3, 'gray')
ax12.axis('off')
# ax12.set_title('Mag Spec 3')

FinalFig.show()
