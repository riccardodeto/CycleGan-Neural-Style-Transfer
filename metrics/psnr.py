import numpy as np
from skimage.metrics import peak_signal_noise_ratio


# Function to calculate the Peak Signal to Noise Ratio (PSNR)
def calculate_psnr(real_images, generated_images):
    psnr_values = []

    # Iterate through each pair of real and generated images
    for i in range(len(real_images)):
        # Calculate PSNR for the current pair of images
        psnr_values.append(peak_signal_noise_ratio(real_images[i], generated_images[i], data_range=255))

    # Calculate the average PSNR value over all pairs of images
    average_psnr = np.mean(psnr_values)
    return average_psnr