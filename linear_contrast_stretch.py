import cv2 as cv
import numpy as np


def linearContrastStretch(img, Smi, Sma):

    Rmin = img.min()
    Rmax = img.max()

    S =  (img - Rmin) * ( (Sma - Smi) / (Rmax - Rmin) ) + Smi

    contrat_stretched_image = S.astype(np.uint8)

    return contrat_stretched_image



image = cv.imread("./pics/low contrast monkey.jpg")

# these two parameters will define the new contrast of the image, and these can be changed according the slider input given by the user on the gui
Smin = 0
Smax = 255


stretchedImage = linearContrastStretch(image, Smin, Smax)

cv.imshow("original_image", image)
cv.imshow("linear_contrast_stretched_image", stretchedImage)
k = cv.waitKey(0)

