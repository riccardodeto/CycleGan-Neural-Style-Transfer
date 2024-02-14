import tensorflow as tf
import os


def load_style_dataset(path, image_size=(256, 256)):
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, image_size)
            img = tf.expand_dims(img, 0)
            images.append(img)  # 0-255

    # Si ha in output un tensore che rappresenta il dataset di stile (immagini di stile concatenate)
    return tf.concat(images, axis=0)