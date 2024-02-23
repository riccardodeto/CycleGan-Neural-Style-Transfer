from data_loading.load_dataset import create_dataset
from preprocessing.preprocess_images import preprocess_dataset
from architecture.models import unet_generator_with_attention, discriminator
import tensorflow as tf
from ckpt_management.ckpt_manager_setup import ckpt_manager_setup
from ckpt_management.restore_checkpoint import restore_checkpoint
from utils.generate_images import generate_images
from utils.save_image import save_image
from utils.save_predictions import save_predictions
from utils.save_metrics import save_metrics
from utils.save_losses import save_losses
from metrics.utils.calculate_mean_gram_matrix import calculate_mean_gram_matrix
from architecture.iterations import Iterations
from metrics.calculate_metrics import calculate_metrics
from ckpt_management.save_checkpoint import save_checkpoint
import time
from tqdm import tqdm
from test.test import test
# from google.colab import drive
# drive.mount('/content/drive')


"""CONSTANTS"""

EPOCHS = 300
CHECKPOINTS_TO_KEEP = 100
TRAINING_SET_SIZE = 1000
VALIDATION_SET_SIZE = 200
TEST_SET_SIZE = 50

CHECKPOINT_INTERVAL = 4
VALIDATION_INTERVAL = 2

BUFFER_SIZE = TRAINING_SET_SIZE
BATCH_SIZE = 20
VAL_BATCH_SIZE = 20
LR_G = 2e-4
LR_D = 2e-4

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

# Load the dataset
dataset, _ = create_dataset()

# Split the dataset
train_portrait, train_naruto = dataset['trainA'], dataset['trainB']
val_portrait, val_naruto = dataset['valA'], dataset['valB']
test_portrait, test_naruto = dataset['testA'], dataset['testB']

# Preprocess the dataset
train_portrait, val_portrait, test_portrait, train_naruto, val_naruto, test_naruto = (preprocess_dataset(
    train_portrait, val_portrait, test_portrait, train_naruto, val_naruto, test_naruto, BUFFER_SIZE, BATCH_SIZE, VAL_BATCH_SIZE))

# Compute the mean gram matrix of style images
mean_matrix_N = calculate_mean_gram_matrix(path="/content/dataset/valB")

# Initialize the generators
generator_pn = unet_generator_with_attention(OUTPUT_CHANNELS, norm_type='instancenorm', attention=True)
generator_np = unet_generator_with_attention(OUTPUT_CHANNELS, norm_type='instancenorm', attention=True)

# Initialize the discriminators
discriminator_p = discriminator(norm_type='instancenorm')
discriminator_n = discriminator(norm_type='instancenorm')

# Print the model architecture
generator_pn.summary()
tf.keras.utils.plot_model(generator_pn, show_shapes=True, dpi=128)

# Initialize optimizers for generators and discriminators
generator_pn_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=0.5)
generator_np_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=0.5)

discriminator_p_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.5)
discriminator_n_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.5)

# Initialize the checkpoint manager
ckpt, ckpt_manager, last_checkpoint_number = ckpt_manager_setup(generator_pn, generator_np, discriminator_p,
                                                                discriminator_n, generator_pn_optimizer,
                                                                generator_np_optimizer, discriminator_p_optimizer,
                                                                discriminator_n_optimizer, CHECKPOINTS_TO_KEEP)


# The Iterations class is responsible for a single step of training
iteration = Iterations(generator_pn, generator_np, discriminator_p, discriminator_n, generator_pn_optimizer,
                       generator_np_optimizer, discriminator_p_optimizer, discriminator_n_optimizer)

# Check if there exists a checkpoint and restore the last checkpoint
initial_epoch = restore_checkpoint(ckpt_manager, ckpt)
print(f'Starting epoch: {initial_epoch}')

sample_portrait = next(iter(train_portrait))
sample_naruto = next(iter(train_naruto))



for epoch in range(initial_epoch, EPOCHS+1):

    """TRAINING"""

    start = time.time()
    # Initialize a progress bar with the total number of iterations
    progress_bar = tqdm(total=int(TRAINING_SET_SIZE / BATCH_SIZE), desc='Training Progress', leave=False)

    # Initialize a vector to accumulate losses (10 items initialized to 0)
    train_losses_sum = [0] * 10

    # Initialize a counter and vectors that will contain the images on which to calculate the metrics
    current_batch = 0
    all_gen_images = []
    all_real_images = []
    all_train_images = []

    # Use tf.data.Dataset.zip() to iterate over two datasets simultaneously
    for image_p, image_n in tf.data.Dataset.zip((train_portrait, train_naruto)):

        # step executes an epoch and returns a list of losses related to that step
        latest_train_losses = iteration.step(image_p, image_n, True)

        # Update the train_losses_sum list by adding up the losses of the current step
        train_losses_sum = [x + y for x, y in zip(train_losses_sum, latest_train_losses)]

        if epoch > 50 or (current_batch % 5) == 0:
            # Save model predictions
            gen_images, real_images, train_images = save_predictions(generator_pn, image_p, image_n, epoch,
                                                                     'train', BATCH_SIZE, current_batch)

            # Every 5 iterations extends the lists with images to calculate training metrics
            if (current_batch % 5) == 0:
                all_gen_images.extend(gen_images)
                all_real_images.extend(real_images)
                all_train_images.extend(train_images)

        current_batch += 1
        progress_bar.update(1)

    progress_bar.close()

    # For each epoch a generated image, the source image and the activation map are generated and saved
    gen_image = generate_images(generator_pn, sample_portrait, discriminator_n)
    save_image(gen_image, epoch, '', '/content/drive/MyDrive/cycleGan/img_predizioni', verbose=False)


    print('\nTraining losses:\n')
    # train_losses_sum: list containing the sum of each loss
    # losses are normalised (average of each loss is saved in train_losses_list)
    train_losses_list = [x / (TRAINING_SET_SIZE / BATCH_SIZE) for x in train_losses_sum]
    save_losses(epoch=epoch, losses=train_losses_list, verbose=True,
                path='/content/drive/MyDrive/cycleGan/loss_train.txt')

    print('\nTraining metrics:\n')
    # Calculate and save all metrics
    train_metrics = calculate_metrics(all_gen_images, all_real_images, all_train_images, style=False, mean_matrix=mean_matrix_N)
    save_metrics(epoch=epoch, metrics=train_metrics, verbose=True,
                 path='/content/drive/MyDrive/cycleGan/metrics_train.txt')

    print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

    """VALIDAZIONE"""

    if (epoch >= 1) and (epoch % VALIDATION_INTERVAL) == 0:
        start = time.time()

        # Initialize a vector that will accumulate losses iteration by iteration
        val_losses_sum = [0] * 10

        val_progress_bar = tqdm(total=VALIDATION_SET_SIZE / VAL_BATCH_SIZE, desc='Validation Progress', leave=False)
        current_batch = 0

        all_gen_images = []
        all_real_images = []
        all_val_images = []

        # Use tf.data.Dataset.zip() to iterate over two datasets simultaneously
        for val_image_p, val_image_n in tf.data.Dataset.zip((val_portrait, val_naruto)):

            # step executes an epoch and returns a list of losses related to that step
            latest_val_losses = iteration.step(val_image_p, val_image_n, False)

            # Update the val_losses_sum list by adding up the losses of the current step
            val_losses_sum = [x + y for x, y in zip(val_losses_sum, latest_val_losses)]

            # Save model predictions
            gen_images, real_images, val_images = save_predictions(generator_pn, val_image_p, val_image_n, epoch,
                                                                   'val', VAL_BATCH_SIZE, current_batch)

            all_gen_images.extend(gen_images)
            all_real_images.extend(real_images)
            all_val_images.extend(val_images)

            current_batch += 1
            val_progress_bar.update(1)

        val_progress_bar.close()

        print('\nValidation losses:\n')
        # val_losses_sum: list containing the sum of each loss
        # losses are normalised (average of each loss is saved in val_losses_list)
        val_losses_list = [x / (VALIDATION_SET_SIZE / VAL_BATCH_SIZE) for x in val_losses_sum]
        save_losses(epoch=epoch, losses=val_losses_list, verbose=True,
                    path='/content/drive/MyDrive/cycleGan/loss_val.txt')

        print('\nValidation metrics:\n')
        # Calculate and save all metrics
        val_metrics = calculate_metrics(all_gen_images, all_real_images, all_val_images, style=True, mean_matrix=mean_matrix_N)
        save_metrics(epoch=epoch, metrics=val_metrics, verbose=True,
                     path='/content/drive/MyDrive/cycleGan/metrics_val.txt')

        print('Time taken for validation epoch {} is {} sec\n'.format(epoch, time.time() - start))


    if (epoch) % CHECKPOINT_INTERVAL == 0:
        start = time.time()
        # Save the new checkpoint
        save_checkpoint(ckpt_manager, epoch, CHECKPOINTS_TO_KEEP)
        print('\nTime taken to save the checkpoint {} is {} sec\n'.format(epoch, time.time() - start))

"""TEST"""

test(ckpt_manager, ckpt, test_portrait, generator_pn, discriminator_n)
