from data_loading.load_dataset import create_dataset
from preprocessing.preprocess_images import preprocess_dataset
from architecture.models import unet_generator_with_attention, discriminator
import tensorflow as tf
from ckpt_management.ckpt_manager_setup import ckpt_manager_setup
from ckpt_management.restore_checkpoint import restore_checkpoint
from utils.save_predictions import save_predictions
from utils.save_metrics import save_metrics
from utils.save_losses import save_losses
from metrics.utils.calculate_mean_gram_matrix import calculate_mean_gram_matrix
from architecture.iterations import Iterations
from metrics.calculate_metrics import calculate_metrics
import time
from tqdm import tqdm

# from google.colab import drive
# drive.mount('/content/drive')


"""CONSTANTS"""

CHECKPOINTS_TO_KEEP = 100
TRAINING_SET_SIZE = 1000
TEST_SET_SIZE = 50

BUFFER_SIZE = TRAINING_SET_SIZE
BATCH_SIZE = 20
TEST_BATCH_SIZE = 10
LR_G = 2e-4
LR_D = 2e-4

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

# Load the dataset
dataset, metadata = create_dataset()

# Split the dataset
train_portrait, train_naruto = dataset['trainA'], dataset['trainB']
val_portrait, val_naruto = dataset['valA'], dataset['valB']
test_portrait, test_naruto = dataset['testA'], dataset['testB']

# Preprocess the dataset
train_portrait, val_portrait, test_portrait, train_naruto, val_naruto, test_naruto = preprocess_dataset(train_portrait, val_portrait, test_portrait, train_naruto, val_naruto, test_naruto, BUFFER_SIZE, BATCH_SIZE, TEST_BATCH_SIZE)

# Compute the mean gram matrix of style images
mean_matrix_N = calculate_mean_gram_matrix(path="/content/dataset/testB")

# Initialize the generators
generator_pn = unet_generator_with_attention(OUTPUT_CHANNELS, norm_type='instancenorm', attention=True)
generator_np = unet_generator_with_attention(OUTPUT_CHANNELS, norm_type='instancenorm', attention=True)

# Initialize the discriminators
discriminator_p = discriminator(norm_type='instancenorm')
discriminator_n = discriminator(norm_type='instancenorm')

# Print the model architecture
# generator_pn.summary()
# tf.keras.utils.plot_model(generator_pn, show_shapes=True, dpi=128)

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
restored_epoch = restore_checkpoint(ckpt_manager, ckpt) - 1
print(f'Testing epoch: {restored_epoch}')

sample_portrait = next(iter(train_portrait))
sample_naruto = next(iter(train_naruto))




"""TEST"""

epoch = restored_epoch

start = time.time()

# Initialize a vector to accumulate losses (10 items initialized to 0)
test_losses_sum = [0] * 10

test_progress_bar = tqdm(total=TEST_SET_SIZE / TEST_BATCH_SIZE, desc='Test Progress', leave=False)

# Initialize a counter and vectors that will contain the images on which to calculate the metrics
current_batch = 0
all_gen_images = []
all_real_images = []
all_test_images = []

# Use tf.data.Dataset.zip() to iterate over two datasets simultaneously
for test_image_p, test_image_n in tf.data.Dataset.zip((test_portrait, test_naruto)):

    # step executes an epoch and returns a list of losses related to that step
    latest_test_losses = iteration.step(test_image_p, test_image_n, False)

    # Update the train_losses_sum list by adding up the losses of the current step
    test_losses_sum = [x + y for x, y in zip(test_losses_sum, latest_test_losses)]

    # Save model predictions
    gen_images, real_images, test_images = save_predictions(generator_pn, test_image_p, test_image_n, epoch,'test', TEST_BATCH_SIZE, current_batch)

    all_gen_images.extend(gen_images)
    all_real_images.extend(real_images)
    all_test_images.extend(test_images)

    current_batch += 1
    test_progress_bar.update(1)

test_progress_bar.close()

print('\nTest losses:\n')
# test_losses_sum: list containing the sum of each loss
# losses are normalised (average of each loss is saved in test_losses_list)
test_losses_list = [x / (TEST_SET_SIZE / TEST_BATCH_SIZE) for x in test_losses_sum]
save_losses(epoch=epoch, losses=test_losses_list, verbose=True,
            path='/content/drive/MyDrive/cycleGan/loss_test.txt')

print('\nTest metrics:\n')
# Calculate and save all metrics
test_metrics = calculate_metrics(all_gen_images, all_real_images, all_test_images, style=True, mean_matrix=mean_matrix_N)
save_metrics(epoch=epoch, metrics=test_metrics, verbose=True,
            path='/content/drive/MyDrive/cycleGan/metrics_test.txt')

print('Note that fid results might be underestimated or overestimated, respectivly,\n'
      'if the number of images in the test set is smaller or greater than the number of images in the validation set.\n')

print('Time taken for test epoch {} is {} sec\n'.format(epoch, time.time() - start))