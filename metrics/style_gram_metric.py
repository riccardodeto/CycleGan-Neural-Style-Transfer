import tensorflow as tf
from utils.style_model import StyleContentModel
from utils.evaluate_gram_matrix import evaluate_gram_matrix


# funzione per calcolare la distanza dalla gram matrix media delle immagini di riferimento
def style_gram_metric(generated_images, mean_matrix):  # 0-255
    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1']

    extractor = StyleContentModel(style_layers, content_layers)

    generated_grams = evaluate_gram_matrix(generated_images, extractor, 'block1_conv1')

    style_gram_metric = 0
    for gram in generated_grams:
        style_gram_metric += tf.reduce_mean(tf.subtract(mean_matrix, gram) ** 2) * 2e-6
    style_gram_metric = style_gram_metric / len(generated_grams)

    return style_gram_metric