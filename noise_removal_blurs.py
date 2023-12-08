import cv2 
import numpy as np

# Read the image
# image = cv2.imread('./pics/house.jpg')


# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def guassian_blur(img):

    guassBlurred = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imshow("guass.", guassBlurred)


def bilateral_blur(img):

    bilateBlurred = cv2.bilateralFilter(img, 9, 90, 30)
    
    cv2.imshow("bilate.", bilateBlurred)


def median_blur(img):

    medianBlurred = cv2.medianBlur(img, 5)  # Adjust the kernel size as needed
    
    cv2.imshow("median.", medianBlurred)




