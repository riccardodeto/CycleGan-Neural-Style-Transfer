import matplotlib.pyplot as plt


def generate_images(generator, test_input, discriminator):
    prediction = generator(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Activation map PatchGan')
    plt.imshow(discriminator(prediction)[0, ..., -1], cmap='RdBu_r')
    plt.show()

    return prediction[0] * 0.5 + 0.5
