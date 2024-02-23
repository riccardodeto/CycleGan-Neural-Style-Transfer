from architecture.models import unet_generator_with_attention, discriminator
import tensorflow as tf
from ckpt_management.ckpt_manager_setup import ckpt_manager_setup
from ckpt_management.restore_checkpoint import restore_checkpoint
from utils.generate_from_path import generate_from_path


# Initializing the two generators
generator_pn = unet_generator_with_attention(3, norm_type='instancenorm', attention=True)
generator_np = unet_generator_with_attention(3, norm_type='instancenorm', attention=True)
# Initializing the two discriminators
discriminator_p = discriminator(norm_type='instancenorm')
discriminator_n = discriminator(norm_type='instancenorm')

# Initializing optimizers for generators and discriminators
generator_pn_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_np_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_p_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_n_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Initializing checkpoint manager
ckpt, ckpt_manager, last_checkpoint_number = ckpt_manager_setup(generator_pn, generator_np,
                                                                discriminator_p, discriminator_n,
                                                                generator_pn_optimizer, generator_np_optimizer,
                                                                discriminator_p_optimizer, discriminator_n_optimizer,
                                                                1)

restored_epoch = restore_checkpoint(ckpt_manager, ckpt) - 1

generate_from_path('PercorsoDaInserire', generator_pn, discriminator_n, epoch=restored_epoch, suffix='suffisso')
