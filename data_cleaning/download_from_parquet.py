# pip install pyarrow
import pyarrow.parquet as pq
import os
from PIL import Image
from io import BytesIO

# `pq.read_table()` per leggere il dataset Parquet e convertirlo in una tabella PyArrow:
parquet_file = '/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/train-00000-of-00002-12944970063701d5.parquet'
table = pq.read_table(parquet_file)

# Conversione in DataFrame
df = table.to_pandas()
print(df.head())

cartella_output = '/Users/riccardodetomaso/Desktop/VARIE/Progetti/resize/image3'
os.makedirs(cartella_output, exist_ok=True)

# Itera sul DataFrame e salva le immagini nella cartella di output
for index, row in df.iterrows():
    try:
        # Estrai i byte dell'immagine dal DataFrame
        image_bytes = row['image']['bytes']

        # Carica l'immagine dai bytes
        image = Image.open(BytesIO(image_bytes))

        # Crea il percorso completo e salva l'immagine
        percorso_immagine = os.path.join(cartella_output, f"immagine_{index}.jpg")
        image.save(percorso_immagine)

        print(f"Immagine {index} salvata con successo.")
    except Exception as e:
        print(f"Errore durante il salvataggio dell'immagine {index}: {e}")
