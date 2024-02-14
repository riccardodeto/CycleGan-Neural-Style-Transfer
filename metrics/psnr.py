import numpy as np
from skimage.metrics import peak_signal_noise_ratio


# funzione per calcolare il Peak Signal to Noise Ratio (PSNR)
def calculate_psnr(real_images, generated_images):
    psnr_values = []
    for i in range(len(real_images)):
        psnr_values.append(peak_signal_noise_ratio(real_images[i], generated_images[i], data_range=255))

    average_psnr = np.mean(psnr_values)
    return average_psnr