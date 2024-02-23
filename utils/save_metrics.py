def save_metrics(epoch=-1, metrics=None, verbose=False, path=None):
    string = (f'Epoch: {epoch}\n '
              f'Inception Score (IS): {metrics[0]}\n '
              f'Frechet Inception Distance (FID)(portrait/generated): {metrics[1]}\n '
              f'Frechet Inception Distance (FID)(naruto/generated): {metrics[2]}\n '
              f'Peak Signal to Noise Ratio (PSNR): {metrics[3]}\n '
              f'Structural Similarity Index (SSI): {metrics[4]}\n '
              f'Style Gram Metric (SGM): {metrics[5]}\n\n')

    # Open a file to save loss values in append mode
    metrics_file = open(path, 'a')

    # Save the loss values to the file
    metrics_file.write(string)
    metrics_file.close()

    if verbose:
        print(string)