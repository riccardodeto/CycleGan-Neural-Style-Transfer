import tensorflow as tf
import os

def is_jpeg(filename):
    # Check if a filename has a JPEG extension
    return filename.lower().endswith(('.jpeg', '.jpg'))

def resize_images_tensorflow(input_folder, output_folder, target_size=(256, 256)):
    # Check if the output folder exists, otherwise create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all images in the input folder
    image_list = [f for f in os.listdir(input_folder) if is_jpeg(f)]

    for image_name in image_list:
        # Path of the input image
        input_image_path = os.path.join(input_folder, image_name)

        # Read the image using TensorFlow and resize it
        img = tf.io.read_file(input_image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        resized_img = tf.image.resize(img, target_size)

        # Convert the image to uint8 type
        resized_img = tf.cast(resized_img, tf.uint8)

        # Create the path for the output image and save it
        output_image_path = os.path.join(output_folder, image_name)
        tf.io.write_file(output_image_path, tf.image.encode_png(resized_img))

if __name__ == "__main__":
    input_folder = "percorso/resize/image"
    output_folder = "percorso/resize/image_resized"

    resize_images_tensorflow(input_folder, output_folder)
