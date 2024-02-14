from PIL import Image
import os

def compress_images(input_folder, output_folder, quality=85):
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Elabora ogni immagine nella cartella di input
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Apri l'immagine
        with Image.open(input_path) as img:
            # Converti l'immagine in modalità RGB
            rgb_img = img.convert("RGB")

            # Comprimi l'immagine e salvala nella cartella di output
            rgb_img.save(output_path, format="JPEG", quality=quality)

if __name__ == "__main__":
    # Specifica la cartella di input e output
    input_folder = "/Users/riccardodetomaso/Desktop/to_jpg"
    output_folder = "/Users/riccardodetomaso/Desktop/to_jpg_lite"

    # Specifica la qualità desiderata (da 0 a 100, dove 100 è la migliore qualità)
    quality = 85

    # Esegui la compressione delle immagini
    compress_images(input_folder, output_folder, quality)

    print("Compressione completata.")
