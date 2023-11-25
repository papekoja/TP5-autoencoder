import numpy as np
import utils.file_readers as file_readers
import os
import matplotlib.pyplot as plt
import models.autoencoder as autoencoder


config_file: dict = file_readers.read_config('config.json')

# Read the font data
font_data = file_readers.read_fonts_h(config_file['data_path'])
font_data = np.array(font_data)

# Use all patterns for training
data_set = np.array(font_data)

print("Loading model...")
ae: autoencoder.AutoEncoder = file_readers.read_model(config_file["load_path"])
# Extracting the file name
file_name = os.path.basename(config_file["load_path"])
print("Autoencoder loaded loaded: ", file_name)

# Collecting latent representations
latent_representations = [ae.encode(pattern) for pattern in data_set]

# Converting to a numpy array for easier plotting
latent_representations = np.array(latent_representations)

# List of characters in the same order as patterns in data_set
characters = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']

# Plotting the latent representations with annotations
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(latent_representations):
    plt.scatter(x, y, marker='o')
    plt.annotate(characters[i], (x, y), textcoords="offset points", xytext=(5,5), ha='center')

plt.title("2D Latent Space Representation with Character Annotations")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.grid(True)
plt.show()
