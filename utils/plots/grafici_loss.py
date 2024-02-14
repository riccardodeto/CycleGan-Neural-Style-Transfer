import re
import matplotlib.pyplot as plt

def parse_input_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pattern = re.compile(r'Epoch: (\d+),\s+Generator G Loss: ([\d.]+),\s+Generator F Loss: ([\d.]+),'
                         r'\s+Discriminator X Loss: ([\d.]+),\s+Discriminator Y Loss: ([\d.]+),'
                         r'\s+Cycle Loss 1: ([\d.]+),\s+Cycle Loss 2: ([\d.]+),'
                         r'\s+Identity Loss 1: ([\d.]+),\s+Identity Loss 2: ([\d.]+),'
                         r'\s+Total Generator G Loss: ([\d.]+),\s+Total Generator F Loss: ([\d.]+)')

    matches = pattern.findall(content)
    epoch_v = []
    gen_g_v = []
    gen_f_v = []
    disc_x_v = []
    disc_y_v = []
    cycle1_v = []
    cycle2_v = []
    identity1_v = []
    identity2_v = []
    total_gen_g_v = []
    total_gen_f_v = []

    for match in matches:
        (epoch, gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, cycle1_loss, cycle2_loss, identity1_loss,
         identity2_loss, total_gen_g_loss, total_gen_f_loss) = match

        epoch_v.append(epoch)
        gen_g_v.append(float(gen_g_loss))
        gen_f_v.append(float(gen_f_loss))
        disc_x_v.append(float(disc_x_loss))
        disc_y_v.append(float(disc_y_loss))
        cycle1_v.append(float(cycle1_loss))
        cycle2_v.append(float(cycle2_loss))
        identity1_v.append(float(identity1_loss))
        identity2_v.append(float(identity2_loss))
        total_gen_g_v.append(float(total_gen_g_loss))
        total_gen_f_v.append(float(total_gen_f_loss))

    epoch_v = [int(elemento) for elemento in epoch_v]

    return epoch_v, gen_g_v, gen_f_v, disc_x_v, disc_y_v, cycle1_v, cycle2_v, identity1_v, identity2_v, total_gen_g_v, total_gen_f_v


epoch_v, gen_g_v, gen_f_v, disc_x_v, disc_y_v, cycle1_v, cycle2_v, identity1_v, identity2_v, total_gen_g_v, total_gen_f_v = (
    parse_input_file('/percorso/loss_values.txt'))


# Grafico 1: Total Generator Loss
plt.figure(figsize=(12, 4))

plt.plot(epoch_v, total_gen_g_v, label='Total Generator G Loss', color='blue')
plt.plot(epoch_v, total_gen_f_v, label='Total Generator F Loss', color='orange')

plt.title('Total Generator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(f'img_predizioni/total_gen_loss.png')
plt.show()

# Grafico 2: Generator Loss
plt.figure(figsize=(12, 8))

plt.plot(epoch_v, cycle1_v, label='Cycle Loss 1', color='red')
plt.plot(epoch_v, cycle2_v, label='Cycle Loss 2', color='green')
plt.plot(epoch_v, identity1_v, label='Identity Loss 1', color='purple')
plt.plot(epoch_v, identity2_v, label='Identity Loss 2', color='brown')
plt.plot(epoch_v, gen_g_v, label='Generator G Loss', color='blue')
plt.plot(epoch_v, gen_f_v, label='Generator F Loss', color='orange')

plt.title('Generator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(f'img_predizioni/gen_combo_loss.png')
plt.show()

# Grafico 3: Discriminator Loss
plt.figure(figsize=(12, 4))

plt.plot(epoch_v, disc_x_v, label='Discriminator X Loss', color='red')
plt.plot(epoch_v, disc_y_v, label='Discriminator Y Loss', color='green')

plt.title('Discriminator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(f'img_predizioni/discriminator_loss.png')
plt.show()

# Grafico 4: Combinato
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(epoch_v, total_gen_g_v, label='Total Generator G Loss', color='blue')
plt.plot(epoch_v, total_gen_f_v, label='Total Generator F Loss', color='orange')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epoch_v, cycle1_v, label='Cycle Loss 1', color='red')
plt.plot(epoch_v, cycle2_v, label='Cycle Loss 2', color='green')
plt.plot(epoch_v, identity1_v, label='Identity Loss 1', color='purple')
plt.plot(epoch_v, identity2_v, label='Identity Loss 2', color='yellow')
plt.plot(epoch_v, gen_g_v, label='Generator G Loss', color='blue')
plt.plot(epoch_v, gen_f_v, label='Generator F Loss', color='orange')
plt.title('Generator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epoch_v, disc_x_v, label='Discriminator X Loss', color='red')
plt.plot(epoch_v, disc_y_v, label='Discriminator Y Loss', color='green')
plt.title('Discriminator Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'img_predizioni/combined_loss.png')
plt.show()



# Grafico 5: Generators + Discriminators Loss
plt.figure(figsize=(12, 4))

plt.plot(epoch_v, total_gen_g_v, label='Total Generator G Loss', color='blue')
plt.plot(epoch_v, total_gen_f_v, label='Total Generator F Loss', color='orange')
plt.plot(epoch_v, disc_x_v, label='Discriminator X Loss', color='red')
plt.plot(epoch_v, disc_y_v, label='Discriminator Y Loss', color='green')


plt.title('Selected Losses over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(f'img_predizioni/discr_gen.png')
plt.show()