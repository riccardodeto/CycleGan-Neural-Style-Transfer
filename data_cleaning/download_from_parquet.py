# pip install pyarrow
import pyarrow.parquet as pq
import os
from PIL import Image
from io import BytesIO

# Read the Parquet dataset
parquet_file = 'percorso/train-00000-of-00002-12944970063701d5.parquet'
table = pq.read_table(parquet_file)

# Convert to DataFrame
df = table.to_pandas()
print(df.head())

cartella_output = 'percorso/output'
os.makedirs(cartella_output, exist_ok=True)

# Iterate over the DataFrame and save images to the output folder
for index, row in df.iterrows():
    # Extract image bytes from the DataFrame
    image_bytes = row['image']['bytes']

    # Load the image from bytes
    image = Image.open(BytesIO(image_bytes))

    # Create the path and save the image
    percorso_immagine = os.path.join(cartella_output, f"immagine_{index}.jpg")
    image.save(percorso_immagine)

    print(f"Immagine {index} salvata con successo.")
