import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from inception_score import calculate_inception_score
from fid import calculate_fid
from psnr import calculate_psnr
from ssi import calculate_ssi
from style_gram_metric import style_gram_metric


def calculate_metrics(gen_images, real_images, ref_images, style, mean_matrix):  # 0-255  # 0-255  # 0-255

    # Carica il modello InceptionV3 pre-addestrato e rimuovi l'ultimo strato completamente connesso (output layer)
    inception_model = InceptionV3(weights="imagenet", include_top=False, pooling='avg', input_shape=(256, 256, 3))

    # InceptionV3 richiede che le immagini siano nel formato RGB con valori compresi tra -1 e 1 quindi le immagini vanno preprocessate
    # La funzione di preprocessing richiede che le immagini siano nel range 0-255
    preproc_gen_images = preprocess_input(np.array(gen_images))
    preproc_real_images = preprocess_input(np.array(real_images))
    preproc_ref_images = preprocess_input(np.array(ref_images))

    inception_score = calculate_inception_score(preproc_gen_images, model=inception_model)
    fid_PG, fid_NG = calculate_fid(preproc_gen_images, preproc_real_images, preproc_ref_images, model=inception_model)
    psnr_value = calculate_psnr(real_images, gen_images)
    ssi_value = calculate_ssi(real_images, gen_images)
    if style:
        style_metric = style_gram_metric(gen_images, mean_matrix)
    else:
        style_metric = -1

    return [inception_score, fid_PG, fid_NG, psnr_value, ssi_value, style_metric]