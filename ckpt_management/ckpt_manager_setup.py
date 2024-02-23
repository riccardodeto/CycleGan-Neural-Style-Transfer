import tensorflow as tf

def ckpt_manager_setup(generator_pn, generator_np, discriminator_p, discriminator_n, generator_pn_optimizer, generator_np_optimizer, discriminator_p_optimizer, discriminator_n_optimizer, CHECKPOINTS_TO_KEEP):
    # Define the path to save the checkpoints
    checkpoint_path = "/content/drive/MyDrive/cycleGan/ckpt_management/train"

    # Create a checkpoint instance with the provided objects
    ckpt = tf.train.Checkpoint(generator_g=generator_pn,
                               generator_f=generator_np,
                               discriminator_x=discriminator_p,
                               discriminator_y=discriminator_n,
                               generator_g_optimizer=generator_pn_optimizer,
                               generator_f_optimizer=generator_np_optimizer,
                               discriminator_x_optimizer=discriminator_p_optimizer,
                               discriminator_y_optimizer=discriminator_n_optimizer)

    # Use a custom instance of CheckpointManager for better control over the saved checkpoints
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=CHECKPOINTS_TO_KEEP)

    # Get the number of the last saved checkpoint if available
    last_checkpoint_number = ckpt_manager.latest_checkpoint.split("-")[-1] if ckpt_manager.latest_checkpoint else 0
    return ckpt, ckpt_manager, last_checkpoint_number