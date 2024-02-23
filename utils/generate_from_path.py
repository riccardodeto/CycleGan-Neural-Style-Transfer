import tensorflow as tf
from preprocessing.preprocess_images import preprocess_image_test
from generate_images import generate_images
from save_image import save_image


def generate_from_path(image_path, generator_pn, discriminator_n, epoch=-1, suffix='trial'):

    # Load the specific image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    # Preprocess the specific image
    image = preprocess_image_test(image, label=None)

    # Expand size to meet batch size
    image = tf.expand_dims(image, axis=0)

    # Perform image generation using the model
    gen_image = generate_images(generator_pn, image, discriminator_n)
    save_image(gen_image, epoch, suffix, '/content/drive/MyDrive/cycleGan/img_from_path', verbose=False)