import re
import matplotlib.pyplot as plt

def parse_input_file(file_path):
    with open(file_path, 'r') as file:
        # Reads the contents of the file and stores it in the content variable
        content = file.read()

    # The model looks for a set of values in the file
    pattern = re.compile(r'Epoch: (\d+),\s+Generator G Loss: ([\d.]+),\s+Generator F Loss: ([\d.]+),'
                         r'\s+Discriminator X Loss: ([\d.]+),\s+Discriminator Y Loss: ([\d.]+),'
                         r'\s+Cycle Loss 1: ([\d.]+),\s+Cycle Loss 2: ([\d.]+),'
                         r'\s+Identity Loss 1: ([\d.]+),\s+Identity Loss 2: ([\d.]+),'
                         r'\s+Total Generator G Loss: ([\d.]+),\s+Total Generator F Loss: ([\d.]+)')

    # Utilize pattern to find all matches in the file content
    matches = pattern.findall(content)

    # Turns the list of tuples into a list of lists
        # for match in matches: Iterate on each tuple within the matches list
        # for item in match: Iterate on each item within the current tuple
    data = [[int(item) if item.isdigit() else float(item) for item in match] for match in matches]

    return data

def plot_losses(epoch_v, *losses, labels=None, title=None, filename=None, figsize=(12,8)):
    plt.figure(figsize=figsize)

    for loss, label in zip(losses, labels):
        plt.plot(epoch_v, loss, label=label)

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if filename:
        plt.savefig(filename)
    plt.show()

# Call the parse_input_file function with the file path to get the data
data = parse_input_file('/percorso/loss_values.txt')

# Unpack the returned data into individual lists for each category
epoch_v, gen_g_v, gen_f_v, disc_x_v, disc_y_v, cycle1_v, cycle2_v, identity1_v, identity2_v, total_gen_g_v, total_gen_f_v = data


# Chart 1: Total Generator Loss
plot_losses(epoch_v, total_gen_g_v, total_gen_f_v, labels=['Total Generator G Loss', 'Total Generator F Loss'],
            title='Total Generator Loss over Epochs', filename='img_predizioni/total_gen_loss.png', figsize=(12, 4))

# Chart 2: Generator Loss
plot_losses(epoch_v, cycle1_v, cycle2_v, identity1_v, identity2_v, gen_g_v, gen_f_v,
            labels=['Cycle Loss 1', 'Cycle Loss 2', 'Identity Loss 1', 'Identity Loss 2', 'Generator G Loss', 'Generator F Loss'],
            title='Generator Loss over Epochs', filename='img_predizioni/gen_combo_loss.png', figsize=(12, 8))

# Chart 3: Discriminator Loss
plot_losses(epoch_v, disc_x_v, disc_y_v, labels=['Discriminator X Loss', 'Discriminator Y Loss'],
            title='Discriminator Loss over Epochs', filename='img_predizioni/discriminator_loss.png', figsize=(12, 4))

# Chart 4: Combined Losses
plot_losses(epoch_v, total_gen_g_v, total_gen_f_v, cycle1_v, cycle2_v, identity1_v, identity2_v, gen_g_v, gen_f_v, disc_x_v, disc_y_v,
            labels=['Total Generator G Loss', 'Total Generator F Loss', 'Cycle Loss 1', 'Cycle Loss 2', 'Identity Loss 1', 'Identity Loss 2', 'Generator G Loss', 'Generator F Loss', 'Discriminator X Loss', 'Discriminator Y Loss'],
            title='Training Loss over Epochs', filename='img_predizioni/combined_loss.png', figsize=(15, 10))

# Chart 5: Selected Losses (Generators + Discriminators)
plot_losses(epoch_v, total_gen_g_v, total_gen_f_v, disc_x_v, disc_y_v,
            labels=['Total Generator G Loss', 'Total Generator F Loss', 'Discriminator X Loss', 'Discriminator Y Loss'],
            title='Selected Losses over Epochs', filename='img_predizioni/discr_gen.png', figsize=(12, 4))