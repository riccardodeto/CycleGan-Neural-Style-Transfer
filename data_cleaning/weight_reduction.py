from PIL import Image
import os

def compress_images(input_folder, output_folder, quality):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the image and convert it to RGB mode
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")

            # Compress the image and save it
            rgb_img.save(output_path, format="JPEG", quality=quality)

if __name__ == "__main__":
    input_folder = "percorso/input"
    output_folder = "percorso/output"


    # Esegui la compressione delle immagini
    compress_images(input_folder, output_folder, quality=85)

    print("Compressione completata!")
