import tensorflow as tf
import os

def is_jpeg(filename):
    return filename.lower().endswith(('.jpeg', '.jpg'))

def resize_images_tensorflow(input_folder, output_folder, target_size=(256, 256)):
    # Assicurati che la cartella di output esista, altrimenti creala
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lista di tutte le immagini nella cartella di input
    image_list = [f for f in os.listdir(input_folder) if is_jpeg(f)]

    for image_name in image_list:
        # Percorso completo dell'immagine di input
        input_image_path = os.path.join(input_folder, image_name)

        # Leggi l'immagine utilizzando TensorFlow
        img = tf.io.read_file(input_image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Ridimensiona l'immagine
        resized_img = tf.image.resize(img, target_size)

        # Converti l'immagine a tipo uint8
        resized_img = tf.cast(resized_img, tf.uint8)

        # Crea il percorso per l'immagine di output nella cartella di output
        output_image_path = os.path.join(output_folder, image_name)

        # Salva l'immagine ridimensionata
        tf.io.write_file(output_image_path, tf.image.encode_png(resized_img))

if __name__ == "__main__":
    input_folder = "/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/image"
    output_folder = "/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/image_resized"

    resize_images_tensorflow(input_folder, output_folder)
