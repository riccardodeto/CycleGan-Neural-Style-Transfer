import tensorflow as tf

def ckpt_manager_setup(generator_pn, generator_np, discriminator_p, discriminator_n, generator_pn_optimizer, generator_np_optimizer, discriminator_p_optimizer, discriminator_n_optimizer, CHECKPOINTS_TO_KEEP):
    """Checkpoints"""
    checkpoint_path = "/content/drive/MyDrive/cycleGan/ckpt_management/train"

    ckpt = tf.train.Checkpoint(generator_g=generator_pn,
                               generator_f=generator_np,
                               discriminator_x=discriminator_p,
                               discriminator_y=discriminator_n,
                               generator_g_optimizer=generator_pn_optimizer,
                               generator_f_optimizer=generator_np_optimizer,
                               discriminator_x_optimizer=discriminator_p_optimizer,
                               discriminator_y_optimizer=discriminator_n_optimizer)

    # Modifica: Utilizza un'istanza custom di CheckpointManager
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=CHECKPOINTS_TO_KEEP)
    # Recupera l'ultimo numero del checkpoint salvato
    last_checkpoint_number = ckpt_manager.latest_checkpoint.split("-")[-1] if ckpt_manager.latest_checkpoint else 0
    return ckpt, ckpt_manager, last_checkpoint_number