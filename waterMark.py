import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def pil_to_opencv(image_pil):
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def opencv_to_pil(image_cv):
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))


def add_watermark_text(img, text):
    watermark_image = img.copy()
    draw = ImageDraw.Draw(watermark_image)

    w, h = img.size
    font_size = min(w, h) // 10  # Adjusted font size for clarity

    font = ImageFont.truetype("fonts/ShortBaby-Mg2w.ttf", font_size)

    # Get the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position for watermark
    x = w - text_width - 10
    y = h - text_height - 10

    # Add the Watermark
    draw.text((x, y), text, fill=(255, 255, 255), font=font, anchor='mm')

    width = watermark_image.size[0]
    height = watermark_image.size[1]

    # return the watermarked image
    return watermark_image, width, height


def add_watermark_image(img, watermark):

    width, height = img.size

    # Resizing the watermark into desired size
    size = (100, 100)
    watermark.thumbnail(size)

    # Set the watermark position, if 0, it will set into top left of image
    x = 0
    y = 0

    # Set the watermark position, if written as below, it will set into bottom right of image
    x = width - 100
    y = height - 100

    # Integrate the image watermark into the watermark position
    img.paste(watermark, (x, y))

    width = img.size[0]
    height = img.size[1]

    return img, width, height
