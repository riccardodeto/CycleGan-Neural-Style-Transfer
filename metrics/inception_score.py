import numpy as np
from scipy.stats import entropy


def calculate_inception_score(images, model, num_splits=10):
    preds = []

    # Divide the images into batches to calculate the Inception Score
    for i in range(num_splits):
        # Selects the current batch of images to be processed and makes predictions about these
        split_images = images[i * (len(images) // num_splits): (i + 1) * (len(images) // num_splits)]

        # Adds predictions to the list
        preds.append(model.predict(split_images))

    # Concatenate the predictions into a single array
    preds = np.concatenate(preds, axis=0)

    # Calculate the marginal distribution (the average of the predictions over all images)
    p_yx = np.mean(preds, axis=0)

    # Calculate the KL divergence for each batch and average
    kl_divergences = []
    for pred in preds:
        # Calculate the KL divergence between the marginal distribution and the distribution for each batch
        kl_divergences.append(entropy(pred.T, p_yx.T))
    kl_divergences = np.mean(kl_divergences)

    # Calculate the Inception Score
    inception_score = np.exp(kl_divergences)

    return inception_score