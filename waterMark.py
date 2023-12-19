import cv2
import numpy as np
from functions import pil_to_opencv, opencv_to_pil


def add_watermark(img, watermark):

    # Read the input and watermark images
    img = pil_to_opencv(img)
    watermark = pil_to_opencv(watermark)

    # scaling both images
    percent_of_scaling = 20
    new_width = int(img.shape[1] * percent_of_scaling/100)
    new_height = int(img.shape[0] * percent_of_scaling/100)
    new_dim = (new_width, new_height)
    resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    wm_scale = 40
    wm_width = int(watermark.shape[1] * wm_scale/100)
    wm_height = int(watermark.shape[0] * wm_scale/100)
    wm_dim = (wm_width, wm_height)

    # to create watermark
    resized_wm = cv2.resize(watermark, wm_dim, interpolation=cv2.INTER_AREA)

    h_img, w_img, _ = resized_img.shape
    center_y = int(h_img/2)
    center_x = int(w_img/2)
    h_wm, w_wm, _ = resized_wm.shape
    top_y = center_y - int(h_wm/2)
    left_x = center_x - int(w_wm/2)
    bottom_y = top_y + h_wm
    right_x = left_x + w_wm

    # Ensure the ROI and resized watermark have the same dimensions
    roi = resized_img[top_y:bottom_y, left_x:right_x]
    resized_wm = cv2.resize(resized_wm, (roi.shape[1], roi.shape[0]))

    # Check if the watermark has an alpha channel (transparency)
    if resized_wm.shape[2] == 4:
        # Extract the alpha channel and merge it with the ROI
        alpha_channel = resized_wm[:, :, 3] / 255.0
        result = cv2.addWeighted(roi, 1, resized_wm[:, :, :3], 0.3, 0)
        resized_img[top_y:bottom_y, left_x:right_x] = result
    else:
        # Merge the ROI and resized watermark directly
        result = cv2.addWeighted(roi, 1, resized_wm, 0.3, 0)
        resized_img[top_y:bottom_y, left_x:right_x] = result

    resized_img = cv2.resize(
        resized_img, (img.shape[1], img.shape[0]))

    resized_img = opencv_to_pil(resized_img)

    return resized_img
