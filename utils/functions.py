import cv2
import numpy as np
from PIL import Image

def pil_to_opencv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def opencv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def process_image(img, operation):
    cv_img = pil_to_opencv(img)
    processed_img = operation(cv_img)
    return opencv_to_pil(processed_img)

def resize_image(img, w, h):
    return process_image(img, lambda cv_img: cv2.resize(cv_img, (w, h)))

def rotate_cropped(img, angle):
    def rotate_cv(cv_img):
        center = tuple(np.array(cv_img.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(cv_img, rotation_matrix, cv_img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return process_image(img, rotate_cv)

def rotate_not_cropped(img, angle):
    def rotate_cv(cv_img):
        center = tuple(np.array(cv_img.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cosine, sine = np.abs(rotation_matrix[0, :2])
        new_width = int((cv_img.shape[0] * sine) + (cv_img.shape[1] * cosine))
        new_height = int((cv_img.shape[0] * cosine) + (cv_img.shape[1] * sine))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        return cv2.warpAffine(cv_img, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return process_image(img, rotate_cv)

def horizontal_flip(img):
    return process_image(img, lambda cv_img: cv2.flip(cv_img, 1))

def vertical_flip(img):
    return process_image(img, lambda cv_img: cv2.flip(cv_img, 0))

def crop_image(img):
    def crop(cv_img):
        # Define the region of interest (ROI)
        x, y, w, h = 100, 50, 200, 150
        cropped_img = cv_img[y:y+h, x:x+w]
        height, width, _ = cv_img.shape
        return cv2.resize(cropped_img, (width, height))
    return process_image(img, crop)

def linearContrastStretch(img, Sma):
    def contrast_stretch(cv_img):
        Rmin = cv_img.min()
        Rmax = cv_img.max()
        Smi = 0
        S = (cv_img - Rmin) * ((Sma - Smi) / (Rmax - Rmin)) + Smi
        return S.astype(np.uint8)
    return process_image(img, contrast_stretch)

def image_brightness(img, bright):
    def adjust_brightness(cv_img):
        # Brightness adjustment
        return cv2.addWeighted(cv_img, 1.0, cv_img, 0, bright)
    return process_image(img, adjust_brightness)

def color_space_transform(img, target_color_space):
    def transform(cv_img):
        if target_color_space == 'RGB':
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif target_color_space == 'HSV':
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        elif target_color_space == 'LAB':
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        else:
            raise ValueError(f"Unsupported target color space: {target_color_space}")
    return process_image(img, transform)

def convert_to_grayscale(img):
    return process_image(img, lambda cv_img: cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))

def gaussian_blur(img, n):
    return process_image(img, lambda cv_img: cv2.GaussianBlur(cv_img, (n, n), 0))

def median_blur(img, n):
    return process_image(img, lambda cv_img: cv2.medianBlur(cv_img, n))