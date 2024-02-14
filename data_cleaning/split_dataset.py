import os
import random
import shutil

def split_images(source_folder, train_folder, test_folder, val_folder, train_ratio=0.8, val_ratio=0.2):
    # Assicurati che le cartelle di destinazione esistano
    for folder in [train_folder, test_folder, val_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Lista dei file presenti nella cartella di origine
    all_images = os.listdir(source_folder)

    # Calcola il numero di immagini per il training, validation e test
    num_train = int(len(all_images) * train_ratio)
    num_val = int(len(all_images) * val_ratio)
    num_test = len(all_images) - num_train - num_val

    # Estrai in modo casuale gli indici delle immagini per training, validation e test
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # Copia le immagini nelle rispettive cartelle di training, validation e test
    for i, image_name in enumerate(all_images):
        source_path = os.path.join(source_folder, image_name)
        if i in train_indices:
            destination_path = os.path.join(train_folder, image_name)
        elif i in val_indices:
            destination_path = os.path.join(val_folder, image_name)
        else:
            destination_path = os.path.join(test_folder, image_name)
        shutil.copyfile(source_path, destination_path)

if __name__ == "__main__":
    # Specifica le cartelle di origine, training, validation e test
    source_folder = "/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/naruto_varie_256"
    train_folder = "/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/trainB"
    val_folder = "/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/valB"
    test_folder = "/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/testB"

    # Suddivide le immagini e copia nelle cartelle di training, validation e test
    split_images(source_folder, train_folder, test_folder, val_folder)
