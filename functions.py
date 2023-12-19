import cv2
import numpy as np
from PIL import Image


def resize_image(img, w, h):

    img = pil_to_opencv(img)

    # Resize the image
    resized_image = cv2.resize(img, (w, h))

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    resized_image = opencv_to_pil(resized_image)

    return resized_image


def rotate_cropped(img, angle):

    img = pil_to_opencv(img)

    height, width, _ = img.shape

    # Get the image center
    center = tuple(np.array(img.shape[1::-1]) / 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        img, rotation_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

    rotated_image = opencv_to_pil(rotated_image)

    return rotated_image


def rotate_not_cropped(img, angle):

    img = pil_to_opencv(img)

    height, width, _ = img.shape

    # Get the image center
    center = tuple(np.array(img.shape[1::-1]) / 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Find the new image dimensions
    cosine = np.abs(rotation_matrix[0, 0])
    sine = np.abs(rotation_matrix[0, 1])
    new_width = int((img.shape[0] * sine) + (img.shape[1] * cosine))
    new_height = int((img.shape[0] * cosine) + (img.shape[1] * sine))

    # Adjust the rotation matrix to keep the entire image
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform the rotation without cropping
    rotated_image = cv2.warpAffine(img, rotation_matrix, (new_width, new_height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

    rotated_image = opencv_to_pil(rotated_image)

    return rotated_image


def horizontal_flip(img):

    img = pil_to_opencv(img)

    # Flip the image horizontally
    horizontal_flipped_image = cv2.flip(img, 1)

    horizontal_flipped_image = cv2.cvtColor(
        horizontal_flipped_image, cv2.COLOR_BGR2RGB)

    horizontal_flipped_image = opencv_to_pil(horizontal_flipped_image)

    return horizontal_flipped_image


def vertical_flip(img):

    img = pil_to_opencv(img)

    vertical_flipped_image = cv2.flip(img, 0)

    vertical_flipped_image = cv2.cvtColor(
        vertical_flipped_image, cv2.COLOR_BGR2RGB)

    vertical_flipped_image = opencv_to_pil(vertical_flipped_image)

    return vertical_flipped_image


def crop_image(img):

    img = pil_to_opencv(img)

    # Define the region of interest (ROI)
    x, y, w, h = 100, 50, 200, 150

    # Crop the image
    cropped_image = img[y:y+h, x:x+w]

    height, width, _ = img.shape
    cropped_image = cv2.resize(cropped_image, (width, height))

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    cropped_image = opencv_to_pil(cropped_image)

    return cropped_image


def linearContrastStretch(img, Sma):

    img = pil_to_opencv(img)

    Rmin = img.min()
    Rmax = img.max()
    Smi = 0

    S = (img - Rmin) * ((Sma - Smi) / (Rmax - Rmin)) + Smi

    contrat_stretched_image = S.astype(np.uint8)

    contrat_stretched_image = cv2.cvtColor(
        contrat_stretched_image, cv2.COLOR_BGR2RGB)

    contrat_stretched_image = opencv_to_pil(contrat_stretched_image)

    return contrat_stretched_image


def image_brightness(img, bright):

    img = pil_to_opencv(img)

    # define the contrast and brightness value
    # Contrast control --> the value is 1.0 here to not change the contrast.
    contrast = 1.0
    brightness = bright  # Brightness control --> (-255 to 255)

    b_image = cv2.addWeighted(src1=img, alpha=contrast,
                              src2=img, beta=0, gamma=brightness)

    b_image = cv2.cvtColor(b_image, cv2.COLOR_BGR2RGB)

    b_image = opencv_to_pil(b_image)

    return b_image


def color_space_transform(img_path, target_color_space):

    img = pil_to_opencv(img_path)

    # Ensure the image is in BGR format (OpenCV's default)
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image should be in BGR format.")

    # Convert the image to the target color space
    if target_color_space == 'RGB':
        transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif target_color_space == 'HSV':
        transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif target_color_space == 'LAB':
        transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    else:
        raise ValueError(
            f"Unsupported target color space: {target_color_space}")

    color_transformed_image = opencv_to_pil(transformed_img)
    return color_transformed_image


def convert_to_grayscale(img):
    img = pil_to_opencv(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img = opencv_to_pil(gray)

    return gray_img


def gaussian_blur(img, n):

    img = pil_to_opencv(img)

    guassBlurred = cv2.GaussianBlur(img, (n, n), 0)

    guassBlurred = opencv_to_pil(guassBlurred)

    return guassBlurred


def median_blur(img, n):

    img = pil_to_opencv(img)

    medianBlurred = cv2.medianBlur(img, n)  # Adjust the kernel size as needed

    medianBlurred = opencv_to_pil(medianBlurred)

    return medianBlurred


# transformations
# image compression
# filters like Gaussian blur, median filter, and canny edge detection.
# feature detection
# logarithmic and power law transforms
# perspective tranformation
# image crop
# morphological operations
# segmentation
# image filtering and enhancement
# watermarking and steganography


# helper functions:

def pil_to_opencv(img):
    np_img = np.array(img)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    return cv_img


def opencv_to_pil(img):

    pil_image = Image.fromarray(img)

    return pil_image


