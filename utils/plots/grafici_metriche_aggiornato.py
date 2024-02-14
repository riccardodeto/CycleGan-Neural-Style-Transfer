import re
import matplotlib.pyplot as plt
import os

def parse_metrics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    metrics = {
        'Epoch': r'Epoch: (\d+),',
        'Inception Score': r'Inception Score: ([\d.]+),',
        'FID(Persone - Naruto Generate)': r'Frechet Inception Distance \(FID\)\(portrait/generated\): ([\d.]+),',
        'FID(Naruto - Naruto Generate)': r'Frechet Inception Distance \(FID\)\(naruto/generated\): ([\d.]+),',
        'PSNR (Contenuto iniziale - Generata)': r'Peak Signal to Noise Ratio \(PSNR\): ([\d.]+),',
        'SSI (Contenuto iniziale - Generata)': r'Structural Similarity Index \(SSI\): ([\d.-]+),',
        'Style Gram Matrix G': r'Style Gram Matrix G: ([\d.]+),',
        'Style Gram Matrix F': r'Style Gram Matrix F: ([\d.]+)',
    }

    parsed_metrics = {key: re.findall(pattern, content) for key, pattern in metrics.items()}
    return parsed_metrics


def save_plot(x, y, label, metric_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    file_name = f"{metric_name.lower().replace(' ', '_')}.png"
    save_path_with_name = os.path.join(save_path, file_name)
    plt.savefig(save_path_with_name)
    plt.close()

    print(f'Il grafico è stato salvato in: {save_path_with_name}')


def main():
    input_files = ['percorso_file/metrics_val.txt']
    save_path = 'percorso_file/metriche'

    all_metrics = {
        'Epoch': [],
        'Inception Score': [],
        'FID(Persone - Naruto Generate)': [],
        'FID(Naruto - Naruto Generate)': [],
        'PSNR (Contenuto iniziale - Generata)': [],
        'SSI (Contenuto iniziale - Generata)': [],
        'Style Gram Matrix G': [],
        'Style Gram Matrix F': [],
    }

    for file_path in input_files:
        parsed_metrics = parse_metrics(file_path)
        for metric, values in parsed_metrics.items():
            all_metrics[metric].extend(values)

    all_epochs = list(map(int, all_metrics['Epoch']))
    for metric in all_metrics:
        if metric != 'Epoch':
            all_metrics[metric] = list(map(float, all_metrics[metric]))
            save_plot(all_epochs, all_metrics[metric], label=metric, metric_name=metric, save_path=save_path)


if __name__ == "__main__":
    main()
