from utils.save_image import save_image
from utils.generate_images import generate_images
from ckpt_management.restore_checkpoint import restore_checkpoint


def test(ckpt_manager, ckpt, test_portrait, generator_pn, discriminator_n, epoch=-1):
    # Find the last checkpoint if there is any and restore it
    restored_epoch = restore_checkpoint(ckpt_manager, ckpt) - 1

    # Run the trained model on the test dataset
    for inp in test_portrait.take(50):
        gen_image = generate_images(generator_pn, inp, discriminator_n)
        save_image(gen_image, epoch, 'test', '/content/drive/MyDrive/cycleGan/img_test', verbose=False)
