import os


def save_checkpoint(ckpt_manager, epoch, checkpoints_to_keep):
    # Save the checkpoint for the current epoch
    ckpt_save_path = ckpt_manager.save(checkpoint_number=epoch)
    print('\nSaving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

    # Check if the number of checkpoints exceeds the specified limit and if so, delete the oldest one
    if len(ckpt_manager.checkpoints) >= checkpoints_to_keep:
        checkpoint_to_remove = ckpt_manager.checkpoints[0]
        ckpt_manager.checkpoints.remove(checkpoint_to_remove)

        files_to_remove = [checkpoint_to_remove + '.index', checkpoint_to_remove + '.data-00000-of-00001']

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)