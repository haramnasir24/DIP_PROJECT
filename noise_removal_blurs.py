import cv2 
import numpy as np
# from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('./pics/house.jpg')

# image = cv2.resize(image, (500, 500))


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# guassian blur
guassBlurred = cv2.GaussianBlur(image, (5, 5), 0)

# bilateral blur
bilateBlurred = cv2.bilateralFilter(image, 9, 90, 30)

# median blur
medianBlurred = cv2.medianBlur(image, 5)  # Adjust the kernel size as needed

# denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

# cv2.imshow("original.", image)
cv2.imshow("guass.", guassBlurred)
cv2.imshow("bilate.", bilateBlurred)
cv2.imshow("median.", medianBlurred)

k = cv2.waitKey(0)



