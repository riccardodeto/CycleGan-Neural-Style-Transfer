from numpy import np
from skimage.metrics import structural_similarity


# funzione per calcolare il Structural Similarity Index (SSI)
def calculate_ssi(real_images, generated_images):
    ssi_values = []
    for i in range(len(real_images)):
        ssi_values.append(structural_similarity(real_images[i], generated_images[i], data_range=255, channel_axis=2))

    average_ssi = np.mean(ssi_values)
    return average_ssi