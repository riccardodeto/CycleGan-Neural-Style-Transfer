import re
import matplotlib.pyplot as plt
import os

def parse_metrics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    epochs = re.findall(r'Epoch: (\d+),', content)
    inception_scores = re.findall(r'Inception Score: ([\d.]+),', content)
    fid_portrait_generated = re.findall(r'Frechet Inception Distance \(FID\)\(portrait/generated\): ([\d.]+),', content)
    fid_naruto_generated = re.findall(r'Frechet Inception Distance \(FID\)\(naruto/generated\): ([\d.]+),', content)
    psnr = re.findall(r'Peak Signal to Noise Ratio \(PSNR\): ([\d.]+),', content)
    ssi = re.findall(r'Structural Similarity Index \(SSI\): ([\d.-]+),', content)
    style_gram_matrix_g = re.findall(r'Style Gram Matrix G: ([\d.]+),', content)
    style_gram_matrix_f = re.findall(r'Style Gram Matrix F: ([\d.]+)', content)

    return epochs, inception_scores, fid_portrait_generated, fid_naruto_generated, psnr, ssi, style_gram_matrix_g, style_gram_matrix_f


def save_plot(x, y, label, metric_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'{label}', marker='o')

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

    all_epochs, all_inception_scores, all_fid_real_generated, all_fid_validation_generated, all_psnr, all_ssi, all_style_gram_matrix_g, all_style_gram_matrix_f = [], [], [], [], [], [], [], []

    for file_path in input_files:
        epochs, inception_scores, fid_real_generated, fid_validation_generated, psnr, ssi, style_gram_matrix_g, style_gram_matrix_f = parse_metrics(file_path)
        all_epochs += epochs
        all_inception_scores += inception_scores
        all_fid_real_generated += fid_real_generated
        all_fid_validation_generated += fid_validation_generated
        all_psnr += psnr
        all_ssi += ssi
        all_style_gram_matrix_g += style_gram_matrix_g
        all_style_gram_matrix_f += style_gram_matrix_f

    all_epochs = list(map(int, all_epochs))
    all_inception_scores = list(map(float, all_inception_scores))
    all_fid_real_generated = list(map(float, all_fid_real_generated))
    all_fid_validation_generated = list(map(float, all_fid_validation_generated))
    all_psnr = list(map(float, all_psnr))
    all_ssi = list(map(float, all_ssi))
    all_style_gram_matrix_g = list(map(float, all_style_gram_matrix_g))
    all_style_gram_matrix_f = list(map(float, all_style_gram_matrix_f))


    save_plot(all_epochs, all_inception_scores, 'Inception Score', 'Inception Score', save_path)
    save_plot(all_epochs, all_fid_real_generated, 'FID(Persone - Naruto Generate)', 'FID(Persone - Naruto Generate)', save_path)
    save_plot(all_epochs, all_fid_validation_generated, 'FID(Naruto - Naruto Generate)', 'FID(Naruto - Naruto Generate)',  save_path)
    save_plot(all_epochs, all_psnr, 'PSNR (Contenuto iniziale - Generata)', 'PSNR (Contenuto iniziale - Generata)',  save_path)
    save_plot(all_epochs, all_ssi, 'SSI (Contenuto iniziale - Generata)', 'SSI (Contenuto iniziale - Generata)',  save_path)
    save_plot(all_epochs, all_style_gram_matrix_g, 'Style Gram Matrix G', 'Style Gram Matrix G', save_path)
    save_plot(all_epochs, all_style_gram_matrix_f, 'Style Gram Matrix F', 'Style Gram Matrix F', save_path)

if __name__ == "__main__":
    main()
