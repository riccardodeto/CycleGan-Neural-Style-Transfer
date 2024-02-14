import numpy as np
from scipy.stats import entropy


def calculate_inception_score(images, model, num_splits=10):
    preds = []

    for i in range(num_splits):
        split_images = images[i * (len(images) // num_splits): (i + 1) * (len(images) // num_splits)]
        preds.append(model.predict(split_images))

    preds = np.concatenate(preds, axis=0)
    # Calculate the marginal distribution (average over samples)
    p_yx = np.mean(preds, axis=0)

    # Calculate the KL divergence for each split and average
    kl_divergences = []
    for pred in preds:
        kl_divergences.append(entropy(pred.T, p_yx.T))
    kl_divergences = np.mean(kl_divergences)

    # Calculate the Inception Score
    inception_score = np.exp(kl_divergences)

    return inception_score