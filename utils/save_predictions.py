from save_image import save_image


def save_predictions(model, images_p, images_n, epoch, suffix, batch_size, current_batch):
    generated_images = []
    real_images = []
    reference_images = []

    for i in range(len(images_p)):
        # Generate prediction using the model
        prediction = model(images_p)  # (-1)-1 GIUSTO

        # Scale the pixel values back to [0, 1] for saving the image (perchè la unet ci
        # restituisce un'immagine di output con valori dei pixel tra -1 e 1 quindi lo normalizza)
        prediction_image = prediction[i] * 0.5 + 0.5  # 0-1

        z = current_batch * batch_size + i + 1
        final_suffix = suffix + str(z)

        save_path = f'/content/drive/MyDrive/cycleGan/{suffix}_img_predizioni/epoch_{epoch}'
        save_image(prediction_image, epoch, final_suffix, save_folder=save_path, verbose=False)

        generated_images.append(prediction_image.numpy() * 255)  # 0-255
        real_images.append(images_p[i].numpy() * 127.5 + 127.5)  # 0-255
        reference_images.append(images_n[i].numpy() * 127.5 + 127.5)  # 0-255

    return generated_images, real_images, reference_images  # 0-255  # 0-255  # 0-255
