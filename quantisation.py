import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# image compression using k means clustering


def quantize_image(img, num_colors):

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Flatten the image array to prepare for KMeans clustering
    # KMeans clustering expects a 2D array where each row represents a data point/pixel
    # each column represents the rgb colors
    pixels = img_array.reshape((-1, 3))

    # Apply KMeans clustering to find representative colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the cluster centers (representative colors)
    # each row corresponds to a cluster center
    # and the columns represent the integer RGB values
    colors = kmeans.cluster_centers_.astype(int)

    # Replace each pixel with its nearest representative color
    quantized_pixels = colors[kmeans.labels_]
    quantized_img_array = quantized_pixels.reshape(img_array.shape)

    # Create a new PIL Image from the quantized array
    quantized_img = Image.fromarray(np.uint8(quantized_img_array))

    return quantized_img
