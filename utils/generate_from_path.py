import tensorflow as tf
from preprocessing.preprocess_images import preprocess_image_test
from generate_images import generate_images
from save_image import save_image


def generate_from_path(image_path, generator_pn, discriminator_n, epoch=-1):

    # Carica l'immagine specifica
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    # Preprocessa l'immagine specifica
    image = preprocess_image_test(image, label=None)

    # Espandi le dimensioni per soddisfare le dimensioni del batch
    image = tf.expand_dims(image, axis=0)

    # Esegui la generazione dell'immagine utilizzando il modello
    generate_images(generator_pn, image, discriminator_n)
    save_image(image, epoch, '', '/content/drive/MyDrive/cycleGan/img_From_path', verbose=False)
