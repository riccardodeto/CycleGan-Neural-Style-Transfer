def restore_checkpoint(ckpt_manager, ckpt):

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

        # Estrai il numero di epoche dal nome del checkpoint
        initial_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) + 1

        print(f'Checkpoint {initial_epoch - 1} restored!!')
    else:
        print('No ckpt_management found.')
        initial_epoch = 1

    return initial_epoch
