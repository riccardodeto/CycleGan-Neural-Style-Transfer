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


"""COSTANTI"""

EPOCHS = 200
CHECKPOINTS_TO_KEEP = 100
TRAINING_SET_SIZE = 1000
VALIDATION_SET_SIZE = 200
TEST_SET_SIZE = 50

CHECKPOINT_INTERVAL = 4
VALIDATION_INTERVAL = 2

BUFFER_SIZE = TRAINING_SET_SIZE
BATCH_SIZE = 1
VAL_BATCH_SIZE = 20
LR_G = 2e-4
LR_D = 2e-4

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

# Carico il dataset
dataset, _ = create_dataset()
# Creo gli split del dataset
train_portrait, train_naruto = dataset['trainA'], dataset['trainB']
val_portrait, val_naruto = dataset['valA'], dataset['valB']
test_portrait, test_naruto = dataset['testA'], dataset['testB']

preprocess_dataset(train_portrait, val_portrait, test_portrait, train_naruto, val_naruto, test_naruto, BUFFER_SIZE, BATCH_SIZE, VAL_BATCH_SIZE)

# Calcolo la gram matrix media delle immagini di stile
mean_matrix_N = calculate_mean_gram_matrix(path="/content/dataset/valB")

# Inizializzo i due generatori
generator_pn = unet_generator_with_attention(OUTPUT_CHANNELS, norm_type='instancenorm', attention=True)
generator_np = unet_generator_with_attention(OUTPUT_CHANNELS, norm_type='instancenorm', attention=True)
# Inizializzo i due discriminatori
discriminator_p = discriminator(norm_type='instancenorm')
discriminator_n = discriminator(norm_type='instancenorm')

# Stampa dell'architettura del modello:
generator_pn.summary()
tf.keras.utils.plot_model(generator_pn, show_shapes=True, dpi=128)

# Inizializzo gli ottimizzatori per i generatori e i discriminatori
generator_pn_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=0.5)
generator_np_optimizer = tf.keras.optimizers.Adam(LR_G, beta_1=0.5)

discriminator_p_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.5)
discriminator_n_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.5)

# Inizializzo il checkpoint manager
ckpt, ckpt_manager, last_checkpoint_number = ckpt_manager_setup(generator_pn, generator_np, discriminator_p, discriminator_n, generator_pn_optimizer, generator_np_optimizer, discriminator_p_optimizer, discriminator_n_optimizer, CHECKPOINTS_TO_KEEP)

iteration = Iterations(generator_pn, generator_np, discriminator_p, discriminator_n, generator_pn_optimizer, generator_np_optimizer, discriminator_p_optimizer, discriminator_n_optimizer)

# Verifico se esiste un checkpoint e ripristina l'ultimo checkpoint
initial_epoch = restore_checkpoint(ckpt_manager, ckpt)
print(f'Starting epoch: {initial_epoch}')

sample_portrait = next(iter(train_portrait))
sample_naruto = next(iter(train_naruto))

for epoch in range(initial_epoch, EPOCHS):

    """TRAINING"""

    start = time.time()
    # Inizializzo una barra di avanzamento con il numero totale di iterazioni
    progress_bar = tqdm(total=int(TRAINING_SET_SIZE / BATCH_SIZE), desc='Training Progress', leave=False)

    # Inizializzo un vettore che accumulerà le loss iterazione per iterazione
    train_losses_sum = [0] * 10
    # Inizializzo un contatore e dei vettori che conterranno le immagini con cui calcolare le metriche relative al training
    current_batch = 0
    all_gen_images = []
    all_real_images = []
    all_train_images = []
    metrics = []

    for image_p, image_n in tf.data.Dataset.zip((train_portrait, train_naruto)):

        latest_train_losses = iteration.train_step(image_p, image_n)
        # Accumulo le loss ad ogni iterazione
        train_losses_sum = [x + y for x, y in zip(train_losses_sum, latest_train_losses)]

        if epoch > 50 or (current_batch % 5) == 0:
            gen_images, real_images, train_images = save_predictions(generator_pn, image_p, image_n, epoch,'train', BATCH_SIZE, current_batch)

            # Ogni 5 iterazioni un'immagine generata viene salvata per calcolare le metriche relative al training
            if (current_batch % 5) == 0:
                all_gen_images.extend(gen_images)
                all_real_images.extend(real_images)
                all_train_images.extend(train_images)

        current_batch += 1
        progress_bar.update(1)

    progress_bar.close()

    gen_image = generate_images(generator_pn, sample_portrait, discriminator_n)
    save_image(gen_image, epoch, '', '/content/drive/MyDrive/cycleGan/img_predizioni', verbose=False)

    print('\nTraining losses:\n')
    train_losses_list = [x / (TRAINING_SET_SIZE / BATCH_SIZE) for x in train_losses_sum]
    save_losses(epoch=epoch, losses=train_losses_list, verbose=True,
                path='/content/drive/MyDrive/cycleGan/loss_train.txt')

    print('\nTraining metrics:\n')
    train_metrics = calculate_metrics(all_gen_images, all_real_images, all_train_images, style=False, mean_matrix=mean_matrix_N)
    save_metrics(epoch=epoch, metrics=train_metrics, verbose=True,
                 path='/content/drive/MyDrive/cycleGan/metrics_train.txt')

    print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

    """VALIDAZIONE"""

    if (epoch >= 1) and (epoch % VALIDATION_INTERVAL) == 0:

        start = time.time()

        # Inizializzo un vettore che accumulerà le loss iterazione per iterazione
        val_losses_sum = [0] * 10

        val_progress_bar = tqdm(total=VALIDATION_SET_SIZE / VAL_BATCH_SIZE, desc='Validation Progress', leave=False)
        current_batch = 0

        all_gen_images = []
        all_real_images = []
        all_val_images = []

        for val_image_p, val_image_n in tf.data.Dataset.zip((val_portrait, val_naruto)):

            latest_val_losses = iteration.val_step(val_image_p, val_image_n)
            # Accumulo le loss ad ogni iterazione
            val_losses_sum = [x + y for x, y in zip(val_losses_sum, latest_val_losses)]

            gen_images, real_images, val_images = save_predictions(generator_pn, val_image_p, val_image_n, epoch,'val', VAL_BATCH_SIZE, current_batch)

            all_gen_images.extend(gen_images)
            all_real_images.extend(real_images)
            all_val_images.extend(val_images)

            current_batch += 1
            val_progress_bar.update(1)

        val_progress_bar.close()

        print('\nValidation losses:\n')
        val_losses_list = [x / (VALIDATION_SET_SIZE / VAL_BATCH_SIZE) for x in val_losses_sum]
        save_losses(epoch=epoch, losses=val_losses_list, verbose=True,
                    path='/content/drive/MyDrive/cycleGan/loss_val.txt')

        print('\nValidation metrics:\n')
        val_metrics = calculate_metrics(all_gen_images, all_real_images, all_val_images, style=True, mean_matrix=mean_matrix_N)
        save_metrics(epoch=epoch, metrics=val_metrics, verbose=True,
                     path='/content/drive/MyDrive/cycleGan/metrics_val.txt')

        print('Time taken for validation epoch {} is {} sec\n'.format(epoch, time.time() - start))

    if (epoch) % CHECKPOINT_INTERVAL == 0:
        start = time.time()
        # Salva il nuovo checkpoint
        save_checkpoint(ckpt_manager, epoch, CHECKPOINTS_TO_KEEP)
        print('\nTime taken to save the checkpoint {} is {} sec\n'.format(epoch, time.time() - start))

"""TEST"""

test(ckpt_manager, ckpt, test_portrait, generator_pn, discriminator_n)
