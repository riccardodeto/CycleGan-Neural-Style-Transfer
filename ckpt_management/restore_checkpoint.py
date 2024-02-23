def restore_checkpoint(ckpt_manager, ckpt):
    # Check if latest checkpoint is available
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

        # Extract the epoch number from the checkpoint name
        initial_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) + 1

        print(f'Checkpoint {initial_epoch - 1} restored!!')
    else:
        print('No checkpoint found.')
        initial_epoch = 1

    return initial_epoch