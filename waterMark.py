import cv2
import numpy as np
from functions import pil_to_opencv, opencv_to_pil


def add_watermark(input_image, watermark_img, transparency=0.5):

    # Read the input and watermark images
    input_image = pil_to_opencv(input_image)
    watermark_image = pil_to_opencv(watermark_img)

    # Check if the watermark image has an alpha channel
    if watermark_image.shape[2] == 4:
        # Resize the watermark image to fit the input image
        h, w, _ = input_image.shape
        watermark_image = cv2.resize(watermark_image, (w, h))

        # Extract the alpha channel (transparency) of the watermark image
        alpha_channel = watermark_image[:, :, 3] / 255.0 * transparency

        # Blend the images
        for c in range(0, 3):
            input_image[:, :, c] = (
                1 - alpha_channel) * input_image[:, :, c] + alpha_channel * watermark_image[:, :, c]

        # input_img = opencv_to_pil(input_image)
        return input_image
    else:
        # If the watermark image does not have an alpha channel, create one
        h, w, _ = input_image.shape
        alpha_channel = np.ones((h, w), dtype=np.uint8) * \
            255  # Fully opaque alpha channel

        # Merge the watermark image with the alpha channel
        watermark_image_with_alpha = cv2.merge(
            [watermark_image, alpha_channel])

        # Resize the watermark image with alpha channel to fit the input image
        watermark_image_with_alpha_resized = cv2.resize(
            watermark_image_with_alpha, (w, h))

        # Ensure that both images have the same depth
        if input_image.dtype != watermark_image_with_alpha_resized.dtype:
            watermark_image_with_alpha_resized = watermark_image_with_alpha_resized.astype(
                input_image.dtype)

        # Blend the images
        input_image = cv2.addWeighted(
            input_image, 1 - transparency, watermark_image_with_alpha_resized, transparency, 0)

        return input_image
