import re
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_metrics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Using regex to extract data from metrics
    data = re.findall(
        r'Epoch: (\d+),.*?Inception Score: ([\d.]+),.*?FID\(portrait/generated\): ([\d.]+),'
        r'.*?FID\(naruto/generated\): ([\d.]+),.*?PSNR: ([\d.]+),.*?SSI: ([\d.-]+),.*?Style Gram Matrix G: ([\d.]+),'
        r'.*?Style Gram Matrix F: ([\d.]+)', content)

    # Creating a pandas DataFrame from the extracted data
    df = pd.DataFrame(data, columns=['Epoch', 'Inception Score', 'FID Portrait Generated', 'FID Naruto Generated',
                                     'PSNR', 'SSI', 'Style Gram Matrix G', 'Style Gram Matrix F'])

    # Converting columns to appropriate types
    df = df.astype(
        {'Epoch': int, 'Inception Score': float, 'FID Portrait Generated': float, 'FID Naruto Generated': float,
         'PSNR': float, 'SSI': float, 'Style Gram Matrix G': float, 'Style Gram Matrix F': float})

    return df


def save_plot(data, column, label, save_path):
    plt.figure(figsize=(8, 6))
    # Generates a plot representing the evolution of metrics in the various epochs
    plt.plot(data['Epoch'], data[column], label=label, marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    # Create the file name and save the plot
    file_name = f"{label.lower().replace(' ', '_')}.png"
    save_path_with_name = os.path.join(save_path, file_name)
    plt.savefig(save_path_with_name)
    plt.close()

    print(f'Il grafico è stato salvato in: {save_path_with_name}')


def main():
    input_file = ['percorso_file/metrics_val.txt']
    save_path = 'percorso_file/metriche'

    # Analyze data from all input files
    all_data = parse_metrics(input_file)

    # Dictionary of metrics to plot
    metrics_to_plot = {'Inception Score': 'Inception Score',
                       'FID Portrait Generated': 'FID(Persone - Naruto Generate)',
                       'FID Naruto Generated': 'FID(Naruto - Naruto Generate)',
                       'PSNR': 'PSNR (Contenuto iniziale - Generata)',
                       'SSI': 'SSI (Contenuto iniziale - Generata)',
                       'Style Gram Matrix G': 'Style Gram Matrix G',
                       'Style Gram Matrix F': 'Style Gram Matrix F'}

    # Plot and save the graphs for each metric
    for column, label in metrics_to_plot.items():
        save_plot(all_data, column, label, save_path)

if __name__ == "__main__":
    main()
