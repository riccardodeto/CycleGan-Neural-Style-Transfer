import cv2
import os
import shutil

def ritaglia_e_salva_immagini(input_directory, output_directory):
    # Elabora ogni file nell'input_directory
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Gestisci errori nella lettura dell'immagine
        try:
            # Leggi l'immagine
            img = cv2.imread(input_path)

            if img is None:
                print(f"Errore nella lettura dell'immagine: {input_path}")
                continue  # Passa alla prossima iterazione del ciclo
        except Exception as e:
            print(f"Errore: {e}")
            continue  # Passa alla prossima iterazione del ciclo

        # Verifica se l'immagine ha un'altezza maggiore della larghezza
        height, width, _ = img.shape
        if height > width:
            # Calcola la nuova altezza
            new_height = width

            # Ritaglia l'immagine (dal basso)
            img_cropped = img[0:new_height, :]

            # Salva l'immagine ritagliata nel nuovo percorso
            cv2.imwrite(output_path, img_cropped)

            print(f"Ritagliata e salvata l'immagine: {output_path}")
        elif width > height:
            # Calcola la nuova dimensione
            new_dimension = height

            # Calcola la quantità da tagliare a destra e sinistra
            cut_amount = (width - new_dimension) // 2

            # Ritaglia l'immagine (taglio uguale a destra e sinistra)
            img_cropped = img[:, cut_amount:cut_amount + new_dimension, :]

            # Salva l'immagine ritagliata nel nuovo percorso
            cv2.imwrite(output_path, img_cropped)

            print(f"Ritagliata e salvata l'immagine: {output_path}")
        else:
            # L'immagine è già quadrata, quindi copiala direttamente nella cartella di output
            shutil.copy2(input_path, output_directory)
            print(f"Copiata l'immagine quadrata: {filename}")

if __name__ == "__main__":
    input_directory = '/Users/riccardodetomaso/Desktop/input'
    output_directory = '/Users/riccardodetomaso/Desktop/output'

    ritaglia_e_salva_immagini(input_directory, output_directory)
