import cv2
from functions import pil_to_opencv
from functions import opencv_to_pil


def apply_otsu_thresholding(img_path):
    # Load the image in grayscale
    imgray= pil_to_opencv(img_path)

    # Convert the image to grayscale
    img = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert NumPy array to PIL Image
    thresholded_pil_image = opencv_to_pil(thresholded_image)

    return thresholded_pil_image
