import os
import random
import shutil

def split_images(source_folder, train_folder, test_folder, val_folder, train_ratio=0.8, val_ratio=0.2):
    # Check if the destination folder exists
    for folder in [train_folder, test_folder, val_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

        # List all files in the source folder
        all_images = os.listdir(source_folder)

        # Calculate the number of images for training, validation, and test
        num_train = int(len(all_images) * train_ratio)
        num_val = int(len(all_images) * val_ratio)

        # Randomly shuffle indices for training, validation, and test images
        indices = list(range(len(all_images)))
        random.shuffle(indices)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Copy images to respective training, validation, and test folders
        for i, image_name in enumerate(all_images):
            source_path = os.path.join(source_folder, image_name)
            destination_path = ''

            if i in train_indices:
                destination_path = os.path.join(train_folder, image_name)

            elif i in val_indices:
                destination_path = os.path.join(val_folder, image_name)

            elif i in test_indices:
                destination_path = os.path.join(test_folder, image_name)

            shutil.copyfile(source_path, destination_path)

    if __name__ == "__main__":
        # Specify source, training, validation, and test folders
        source_folder = "path/naruto_256"
        train_folder = "path/resize/trainB"
        val_folder = "path/resize/valB"
        test_folder = "path/resize/testB"

        # Split images and copy them to training, validation, and test folders
        split_images(source_folder, train_folder, test_folder, val_folder)
