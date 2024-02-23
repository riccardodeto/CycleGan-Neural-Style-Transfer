import numpy as np
from skimage.metrics import structural_similarity


# Function to calculate the Structural Similarity Index (SSI)
def calculate_ssi(real_images, generated_images):
    ssi_values = []

    # Iterate through each pair of real and generated images
    for i in range(len(real_images)):
        # Calculate SSI for the current pair of images
        # `data_range`:specifies the range of the input values (0-255 for uint8 images)
        # `channel_axis`: specifies the axis representing color channels (2 for RGB images)
        ssi_values.append(structural_similarity(real_images[i], generated_images[i], data_range=255, channel_axis=2))

    # Calculate the average SSI value over all pairs of images
    average_ssi = np.mean(ssi_values)
    return average_ssi