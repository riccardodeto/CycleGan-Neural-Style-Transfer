import os


def save_checkpoint(ckpt_manager, epoch, checkpoints_to_keep):
    ckpt_save_path = ckpt_manager.save(checkpoint_number=epoch)
    print('\nSaving checkpoint for epoch {} at {}'.format(epoch,
                                                          ckpt_save_path))

    if len(ckpt_manager.checkpoints) >= checkpoints_to_keep:
        checkpoint_to_remove = ckpt_manager.checkpoints[0]
        ckpt_manager.checkpoints.remove(checkpoint_to_remove)

        index_file_path = checkpoint_to_remove + '.index'
        data_file_path = checkpoint_to_remove + '.data-00000-of-00001'

        if os.path.exists(index_file_path):
            os.remove(index_file_path)