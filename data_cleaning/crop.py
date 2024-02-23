import cv2
import os
import shutil

def cut_and_save_img(input_directory, output_directory):
    # Process each file in the input_directory
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Read the image
        img = cv2.imread(input_path)


        # Check if the image has greater height than width
        height, width, _ = img.shape
        if height > width:
            # Calculate the new height
            new_height = width

            # Crop the image (from bottom)
            img_cropped = img[0:new_height, :]

            # Save the cropped image
            cv2.imwrite(output_path, img_cropped)

            print(f"Ritagliata e salvata l'immagine: {output_path}")


        elif width > height:
            # Calculate the new dimension
            new_dimension = height

            # Calculate the amount to cut on both sides
            cut_amount = (width - new_dimension) // 2

            # Crop the image (equal cut on both sides)
            img_cropped = img[:, cut_amount:cut_amount + new_dimension, :]

            # Save the cropped image
            cv2.imwrite(output_path, img_cropped)

            print(f"Image cropped and saved: {output_path}")

        else:
            # The image is already square
            shutil.copy2(input_path, output_directory)
            print(f"Copiata l'immagine quadrata: {filename}")

if __name__ == "__main__":
    input_directory = 'percorso/input'
    output_directory = 'percorso/output'

    cut_and_save_img(input_directory, output_directory)
