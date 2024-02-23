def save_losses(epoch=-1, losses=None, verbose=False, path=None):
    string = (f'Epoch: {epoch}\n '
              f'Generator PN Loss: {losses[0]}\n '
              f'Generator NP Loss: {losses[1]}\n '
              f'Discriminator P Loss: {losses[2]}\n '
              f'Discriminator N Loss: {losses[3]}\n '
              f'Cycle Loss 1: {losses[4]}\n '
              f'Cycle Loss 2: {losses[5]}\n '
              f'Identity Loss 1: {losses[6]}\n '
              f'Identity Loss 2: {losses[7]}\n '
              f'Total Generator PN Loss: {losses[8]}\n '
              f'Total Generator NP Loss: {losses[9]}\n\n')

    # Open a file to save loss values in append mode
    loss_file = open(path, 'a')

    # Save the loss values to the file
    loss_file.write(string)
    loss_file.close()

    if verbose:
        print(string)