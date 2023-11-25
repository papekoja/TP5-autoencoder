import utils.file_readers as file_readers
import os
import models.autoencoder as autoencoder
import numpy as np
import matplotlib.pyplot as plt

config_file: dict = file_readers.read_config('config.json')

print("Loading model...")
ae: autoencoder.AutoEncoder = file_readers.read_model(config_file["load_path"])
# Extracting the file name
file_name = os.path.basename(config_file["load_path"])
print("Autoencoder loaded loaded: ", file_name)


def get_user_input():
    """ Prompt user for two coordinates """
    x = float(input("Enter the first coordinate (x): "))
    y = float(input("Enter the second coordinate (y): "))
    return np.array([[x, y]])


def generate_image(latent_vector):
    """ Generate an image from the decoder given a latent vector """
    decoded = ae.decode(latent_vector)
    return decoded.reshape((7, 5))  # Reshape according to your image dimensions


while True:
    # Get user input
    latent_vector = get_user_input()

    # Generate image
    image = generate_image(latent_vector)

    # Display the image
    plt.imshow(image, cmap='gray')  # Adjust color map or parameters as needed
    plt.axis('off')
    plt.show()
