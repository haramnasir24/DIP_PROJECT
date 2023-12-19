import cv2
import numpy as np
from PIL import Image



def resize_image(img, new_width, new_height):

    img = pil_to_opencv(img)

    # Get the dimensions of the original image
    height, width = img.shape[:2]

    # Create an empty array for the resized image
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate scaling factors for height and width
    scale_y = height / new_height
    scale_x = width / new_width

    # Iterate over each pixel in the resized image
    for y in range(new_height):
        for x in range(new_width):
            # Calculate the corresponding position in the original image
            original_y = y * scale_y
            original_x = x * scale_x

            # Find the four nearest pixels in the original image
            y1 = int(original_y)
            y2 = min(y1 + 1, height - 1)
            x1 = int(original_x)
            x2 = min(x1 + 1, width - 1)

            # Bilinear interpolation
            dy = original_y - y1
            dx = original_x - x1

            # Interpolate values for each channel
            interpolated_pixel = (
                (1 - dx) * (1 - dy) * img[y1, x1] +
                dx * (1 - dy) * img[y1, x2] +
                (1 - dx) * dy * img[y2, x1] +
                dx * dy * img[y2, x2]
            )

            # Set the pixel value in the resized image
            resized_image[y, x] = np.round(interpolated_pixel).astype(np.uint8)

    resized_image = opencv_to_pil(resized_image)

    return resized_image


def rotate_cropped(img, angle):

    img = pil_to_opencv(img)

    # Get the image center
    center = tuple(np.array(img.shape[1::-1]) / 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, rotation_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)


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
    rotated_image = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


    rotated_image = opencv_to_pil(rotated_image)

    return rotated_image


def horizontal_flip(img):

    img = pil_to_opencv(img)

    # horizontal_flipped_image = cv2.flip(img, 1)

    height, width = img.shape[:2]

    horizontal_flipped_image = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            horizontal_flipped_image[i, j] = img[i, width - 1 - j]

    horizontal_flipped_image = opencv_to_pil(horizontal_flipped_image)

    return horizontal_flipped_image


def vertical_flip(img):

    img = pil_to_opencv(img)

    # vertical_flipped_image = cv2.flip(img, 0)

    height, width = img.shape[:2]

    vertical_flipped_image = np.zeros_like(img)

    for i in range(height):
        for j in range(width):
            vertical_flipped_image[i, j] = img[height - 1 - i, j]


    vertical_flipped_image = opencv_to_pil(vertical_flipped_image)

    return vertical_flipped_image


def crop_image(img):

    img = pil_to_opencv(img)

    # Define the region of interest (ROI)
    x, y, w, h = 100, 50, 200, 150

    # Crop the image
    cropped_image = img[y:y+h, x:x+w]

    height, width = img.shape[:2]

    cropped_image = cv2.resize(cropped_image, (width, height))

    cropped_image = opencv_to_pil(cropped_image)

    return cropped_image


def linearContrastStretch(img, Sma):

    img = pil_to_opencv(img)

    Rmin = img.min()
    Rmax = img.max()
    Smi = 0

    S =  (img - Rmin) * ( (Sma - Smi) / (Rmax - Rmin) ) + Smi

    contrat_stretched_image = S.astype(np.uint8)


    contrat_stretched_image = opencv_to_pil(contrat_stretched_image)

    return contrat_stretched_image


def image_brightness(img, bright):

    img = pil_to_opencv(img)

    # define the contrast and brightness value
    contrast = 1.0 # Contrast control --> the value is 1.0 here to not change the contrast.
    brightness = bright # Brightness control --> (-255 to 255)

    b_image = cv2.addWeighted( src1=img, alpha=contrast, src2=img, beta=0, gamma=brightness)

    b_image = opencv_to_pil(b_image)

    return b_image


def guassian_blur(img, n):

    img = pil_to_opencv(img)

    guassBlurred = cv2.GaussianBlur(img, (n, n), 0)

    guassBlurred = opencv_to_pil(guassBlurred)

    return guassBlurred

    


# def bilateral_blur(img):

#     img = pil_to_opencv(img)

#     bilateBlurred = cv2.bilateralFilter(img, 9, 90, 30)

#     bilateBlurred = opencv_to_pil(bilateBlurred)

#     return bilateBlurred
    
    


def median_blur(img, n):

    img = pil_to_opencv(img)

    medianBlurred = cv2.medianBlur(img, n)  # Adjust the kernel size as needed

    medianBlurred = opencv_to_pil(medianBlurred)

    return medianBlurred
    
    

def rgb_to_grayscale(img):

    img = pil_to_opencv(img)

    # Ensure the image is in the correct format
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input must be a 3-channel BGR image")

    # Extract the BGR channels
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    # Calculate luminance
    grayscale_image = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert to uint8 (8-bit) data type
    grayscale_image = grayscale_image.astype(np.uint8)

    grayscale_image = opencv_to_pil(grayscale_image)

    return grayscale_image






# helper functions:

def pil_to_opencv(img):

    np_img = np.array(img)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    return cv_img


def opencv_to_pil(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(img)

    return pil_image

