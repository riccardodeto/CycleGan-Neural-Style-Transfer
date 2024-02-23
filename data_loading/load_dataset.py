import csv
import os
import tensorflow_datasets as tfds


def create_csv(dataset_directory, subfolders):
    # Open the CSV file in write mode
    csv_file_path = '/content/dataset/metadata.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Define the CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the header
        csv_writer.writerow(['image_id', 'domain', 'split', 'image_path'])

        # Iterate through the subfolders
        for split in subfolders:
            folder_path = os.path.join(dataset_directory, split)

            # Find all images in the subfolder
            image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # Write the information to the CSV file
            for image_id in image_files:
                domain = 'A (Ritratti)' if split in ['trainA', 'testA', 'valA'] else 'B (Naruto)'
                image_path = os.path.join(split, image_id)
                csv_writer.writerow([image_id, domain, split, image_path])

    print(f"File CSV creato con successo: {csv_file_path}")


def create_dataset():
    """Load Dataset

        Reads the content of 6 folders and generates a metadata.csv file containing a table with dataset information:
        1. image_id (nome immagine)
        2. domain (A (Persone) oppure B (Naruto))
        3. split (trainA, validationA, testA, trainB, validationB, testB)
        4. image_path (split/image_id)
        """

    # !git clone https://github.com/riccardodeto/dataset.git

    dataset_directory = '/content/dataset'

    subfolders = ['trainA', 'trainB', 'testA', 'testB', 'valA', 'valB']

    # Create the CSV
    create_csv(dataset_directory, subfolders)

    # Load the dataset created in TensorFlow
    dataset, metadata = tfds.load('portraits_naruto', with_info=True, as_supervised=True)
    return dataset, metadata
