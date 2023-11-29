import cv2
import numpy as np

# Read the image
image = cv2.imread('./pics/house.jpg')

# Define the new dimensions
width, height = 300, 200

# Resize the image
resized_image = cv2.resize(image, (width, height))

# Display the original and resized image
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)


# Define the angle of rotation
angle = 90

height, width, _ = image.shape

# Get the image center
center = tuple(np.array(image.shape[1::-1]) / 2)

# Perform the rotation
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

# rotated_image = rotated_image[width, height]

cv2.imshow('Rotated Image', rotated_image)



# Find the new image dimensions
cosine = np.abs(rotation_matrix[0, 0])
sine = np.abs(rotation_matrix[0, 1])
new_width = int((image.shape[0] * sine) + (image.shape[1] * cosine))
new_height = int((image.shape[0] * cosine) + (image.shape[1] * sine))

# Adjust the rotation matrix to keep the entire image
rotation_matrix[0, 2] += (new_width / 2) - center[0]
rotation_matrix[1, 2] += (new_height / 2) - center[1]

# Perform the rotation without cropping
rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


cv2.imshow('Rotated Image (No Crop)', rotated_image)




# Flip the image horizontally
horizontal_flipped_image = cv2.flip(image, 1)

vertical_flipped_image = cv2.flip(image, 0)


cv2.imshow('Vertically Flipped Image', vertical_flipped_image)
cv2.imshow('Horizontally Flipped Image', horizontal_flipped_image)



# Define the region of interest (ROI)
x, y, w, h = 100, 50, 200, 150

# Crop the image
cropped_image = image[y:y+h, x:x+w]

height, width, _ = image.shape
cropped_image = cv2.resize(cropped_image, (width, height))

cv2.imshow('Cropped Image', cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()




