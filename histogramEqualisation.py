import numpy as np
import cv2
from PIL import Image
from functions import pil_to_opencv


def histogram_equalization(img):
    # Read the image
    # Convert the image to a NumPy array

    img_array = pil_to_opencv(img)

    # Convert the image array to grayscale
    if len(img_array.shape) == 3:  # Check if the image is in color
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Get the shape of the image
    x, y = img_array.shape

    # Calculate the histogram
    H = np.zeros(256)
    for i in range(x):
        for j in range(y):
            H[img_array[i, j]] += 1

    # Calculate the cumulative distribution function (CDF)
    CDF = np.cumsum(H)

    # Perform histogram equalization
    equalized_img_array = (CDF[img_array] / CDF.max() * 255).astype(np.uint8)

    # Create a new PIL Image from the equalized array
    equalized_img = Image.fromarray(equalized_img_array)

    return equalized_img
