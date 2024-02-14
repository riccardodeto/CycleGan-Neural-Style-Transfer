import csv
import os
import tensorflow_datasets as tfds

def create_csv(dataset_directory, subfolders):
    # Apri il file CSV in modalità scrittura
    csv_file_path = '/content/dataset/metadata.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Definisci il writer CSV
        csv_writer = csv.writer(csv_file)

        # Scrivi l'intestazione
        csv_writer.writerow(['image_id', 'domain', 'split', 'image_path'])

        # Itera attraverso le sottocartelle
        for split in subfolders:
            folder_path = os.path.join(dataset_directory, split)

            # Trova tutte le immagini nella sottocartella
            image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # Scrivi le informazioni nel file CSV
            for image_id in image_files:
                domain = 'A (Ritratti)' if split in ['trainA', 'testA', 'valA'] else 'B (Naruto)'
                image_path = os.path.join(split, image_id)
                csv_writer.writerow([image_id, domain, split, image_path])

    print(f"File CSV creato con successo: {csv_file_path}")


def create_dataset():
    """Caricamento dataset

    Legge il contenuto delle 6 cartelle e genera un file chiamato metadata.csv contenente una tabella in cui sono fornite le informazioni relative al nostro dataset, tra cui:
    1. image_id (nome immagine)
    2. domain ( A (Persone) oppure B (Naruto) )
    3. split (trainA, validationA, testA, trainB, validationB, testB)
    4. image_path (split/image_id)
    """
    # !git clone https://github.com/riccardodeto/dataset.git
    # Percorso della directory contenente le sottocartelle con gli esempi
    dataset_directory = '/content/dataset'
    # Lista delle sottocartelle
    subfolders = ['trainA', 'trainB', 'testA', 'testB', 'valA', 'valB']
    # Crea il csv
    create_csv(dataset_directory, subfolders)
    # Caricamento del dataset creato in tensorflow e assegnazione di nomi alle varie sottoclassi del dataset
    dataset, metadata = tfds.load('portraits_naruto', with_info=True, as_supervised=True)
    return dataset, metadata
