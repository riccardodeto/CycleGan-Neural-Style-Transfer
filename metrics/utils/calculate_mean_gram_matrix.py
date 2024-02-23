import tensorflow as tf
from load_reference_dataset import load_style_dataset
from style_model import StyleContentModel
from evaluate_gram_matrix import evaluate_gram_matrix

def calculate_mean_gram_matrix(path):
    # Load the style dataset
    style_dataset = load_style_dataset(path)

    # Define the layers for content and style extraction
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Create an instance of the StyleContentModel to calculate gram matrices
    extractor = StyleContentModel(style_layers, content_layers)

    # Calculate all gram matrices for the style dataset
    style_reference_grams = evaluate_gram_matrix(style_dataset, extractor, 'block1_conv1')

    # Calculate the mean of the gram matrices
    mean_matrix = tf.reduce_mean(style_reference_grams, axis=0)

    return mean_matrix