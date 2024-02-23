import tensorflow as tf
from utils.style_model import StyleContentModel
from utils.evaluate_gram_matrix import evaluate_gram_matrix


# Function to calculate the distance from the mean gram matrix of reference images
def style_gram_metric(generated_images, mean_matrix):  # 0-255
    # Define the layers for content and style extraction
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1']

    # Instance of the StyleContentModel to extract style and content features
    extractor = StyleContentModel(style_layers, content_layers)

    # Evaluate the gram matrices of the generated images
    generated_grams = evaluate_gram_matrix(generated_images, extractor, 'block1_conv1')

    style_gram_metric = 0

    # Iterate through each gram matrix of the generated images
    for gram in generated_grams:
        # Calculate the mean squared difference between the mean matrix and the current gram matrix
        style_gram_metric += tf.reduce_mean(tf.subtract(mean_matrix, gram) ** 2) * 2e-6

    # Normalize the style gram metric by the number of generated gram matrices
    style_gram_metric = style_gram_metric / len(generated_grams)

    return style_gram_metric