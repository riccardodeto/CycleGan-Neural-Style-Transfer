import os
import matplotlib.pyplot as plt


def save_image(prediction_image, epoch, suffix, save_folder, verbose=False):
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Save the prediction image
    filename = f"predizione_epoca_{epoch}_{suffix}.png"
    save_path = os.path.join(save_folder, filename)

    # Save the image using matplotlib
    plt.imsave(save_path, prediction_image.numpy())

    if verbose:
        print(f"Prediction saved at: {save_path}")
