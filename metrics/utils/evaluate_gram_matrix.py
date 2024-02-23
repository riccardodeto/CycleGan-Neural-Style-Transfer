import tensorflow as tf


def evaluate_gram_matrix(style_dataset, extractor, layer):
    style_reference_grams = []

    for image in style_dataset:
        # Add a new axis to the image tensor to match the model input shape
        style_image = image[tf.newaxis, :]

        # Extract style features from the image using the specified layer
        style_targets = extractor(style_image)['style']

        # Append the gram matrix of the specified layer to the list
        style_reference_grams.append(style_targets[layer])

    return style_reference_grams