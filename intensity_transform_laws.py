import numpy as np
from PIL import Image


def log_transform(img, c):

    img = np.array(img)

    # Apply log transform.
    # c = 255/(np.log(1 + np.max(img)))

    log_transformed_image = c * np.log(1 + img)

    # Specify the data type.
    log_transformed_image = np.array(log_transformed_image, dtype=np.uint8)

    log_transformed_image = Image.fromarray(log_transformed_image)

    return log_transformed_image


def power_law_transform(img, g):

    img = np.array(img)

    # Apply gamma correction.
    gamma_corrected_image = np.array(255*(img / 255) ** g, dtype='uint8')

    gamma_corrected_image = Image.fromarray(gamma_corrected_image)

    return gamma_corrected_image


def negative_of_image(img):

    img = np.array(img)

    # max_value = max.any(img)

    # negative_image = max_value - img

    negative_image = 255 - img

    negative_image = Image.fromarray(negative_image)

    return negative_image
