import tensorflow as tf
from load_reference_dataset import load_style_dataset
from style_model import StyleContentModel
from evaluate_gram_matrix import evaluate_gram_matrix

def calculate_mean_gram_matrix(path):
    # Load style datasets
    style_dataset = load_style_dataset(path)

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Definiamo un oggetto che ci permetterà di calcolare le gram matrix
    extractor = StyleContentModel(style_layers, content_layers)

    # Calcoliamo tutte le gram matrix del dataset
    style_reference_grams = evaluate_gram_matrix(style_dataset, extractor, 'block1_conv1')

    # Calcolo delle medie delle matrici di Gram
    mean_matrix = tf.reduce_mean(style_reference_grams, axis=0)

    return mean_matrix