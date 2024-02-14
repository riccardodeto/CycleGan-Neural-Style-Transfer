# !pip install pytorch-fid
from pytorch_fid import fid_score
import numpy as np


def calculate_fid(generated_images, real_images, val_images, model):
    # calculate activations
    generated_pred = model.predict(generated_images)
    real_pred = model.predict(real_images)
    val_pred = model.predict(val_images)

    generated_pred = generated_pred[:, :200]
    real_pred = real_pred[:, :200]
    val_pred = val_pred[:, :200]

    mu_generated, sigma_generated = generated_pred.mean(axis=0), np.cov(generated_pred, rowvar=False)
    mu_real, sigma_real = real_pred.mean(axis=0), np.cov(real_pred, rowvar=False)
    mu_val, sigma_val = val_pred.mean(axis=0), np.cov(val_pred, rowvar=False)

    fid_real = fid_score.calculate_frechet_distance(mu_generated, sigma_generated, mu_real, sigma_real)
    fid_val = fid_score.calculate_frechet_distance(mu_generated, sigma_generated, mu_val, sigma_val)
    return fid_real, fid_val