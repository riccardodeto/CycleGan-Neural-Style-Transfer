import tensorflow as tf


def evaluate_gram_matrix(style_dataset, extractor, layer):
    style_reference_grams = []

    for image in style_dataset:
        style_image = image[tf.newaxis, :]
        style_targets = extractor(style_image)['style']
        style_reference_grams.append(style_targets[layer])

    return style_reference_grams