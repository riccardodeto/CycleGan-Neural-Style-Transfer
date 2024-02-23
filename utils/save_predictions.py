from save_image import save_image


def save_predictions(model, images_p, images_n, epoch, suffix, batch_size, current_batch):
    # Lists to store generated images, real images and reference images
    generated_images = []
    real_images = []
    reference_images = []

    for i in range(len(images_p)):
        # Generate prediction using the model
        prediction = model(images_p)

        # Scale the pixel values back to [0, 1] to save the image (UNet returns output in range [-1,1])
        prediction_image = prediction[i] * 0.5 + 0.5  # 0-1

        # Calculate an index for each predicted image
        z = current_batch * batch_size + i + 1
        final_suffix = suffix + str(z)

        # Save the predicted image
        save_path = f'/content/drive/MyDrive/cycleGan/{suffix}_img_predizioni/epoch_{epoch}'
        save_image(prediction_image, epoch, final_suffix, save_folder=save_path, verbose=False)

        # Append the generated, real, and reference images to their respective lists
        generated_images.append(prediction_image.numpy() * 255)  # 0-255
        real_images.append(images_p[i].numpy() * 127.5 + 127.5)  # 0-255
        reference_images.append(images_n[i].numpy() * 127.5 + 127.5)  # 0-255

    # Return the generated images, real images, and reference images
    return generated_images, real_images, reference_images
