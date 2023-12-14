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


# transformations
# image compression
# segmentation
# image filtering and enhancement
# morphological operations
# watermarking and steganography
# color space transformations
# grayscale conversion
# filters like Gaussian blur, median filter, and canny edge detection.
# histogram equalisation
# feature detection
# logarithmic and power law transforms
# perspective tranformation



# helper functions:

def pil_to_opencv(img):

    np_img = np.array(img)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    return cv_img


def opencv_to_pil(img):

    pil_image = Image.fromarray(img)

    return pil_image
