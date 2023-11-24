import numpy as np
import matplotlib.pyplot as plt
import utils.file_readers as file_readers
import utils.file_writer as file_writer
import utils.activation_functions as activation_functions
import models.autoencoder as autoencoder

# Read the font data
config_file: dict = file_readers.read_config('config.json')

# Read the font data
font_data = file_readers.read_fonts_h(config_file['data_path'])
font_data = np.array(font_data)

# Initializes the auto-encoder
activation_function, derived_activation_function = activation_functions.get_activation_functions(
    config_file["activation_function"], config_file["beta"])
auto_encoder = autoencoder.AutoEncoder(activation_function, derived_activation_function, len(font_data[0]),
                                       config_file["hidden_layout"], config_file["latent_dim"],
                                       config_file["momentum"], config_file["momentum_alpha"])

# Use all patterns for training
data_set = np.array(font_data)

if config_file["load"]:
    print("Loading model...")
    auto_encoder = file_readers.read_model(config_file["load_path"])

if config_file["train"]:
    # Train auto-encoder
    for epoch in range(config_file["epochs"]):
        for data in data_set:
            auto_encoder.train(data, data, config_file["learning_rate"])
        auto_encoder.update_weights()
        error = auto_encoder.compute_error(data_set, data_set)
        if error < config_file["error_threshold"]:
            break
        if epoch % 50 == 0:
            print(f'Iteration {epoch}, error {error}')

    # Save the model
    if config_file["save"]:
        file_writer.save_model_and_config(auto_encoder, error, config_file, config_file["save_path"])


reproduced_patterns = []

for pattern in data_set:
    reproduced_patterns.append(auto_encoder.forward_pass(pattern))

# Determine the number of patterns
num_patterns = len(data_set)

# Create a figure with 2 rows and num_patterns columns
plt.figure(figsize=(num_patterns * 2, 4))

for i in range(num_patterns):
    # Process the original pattern
    original_pattern_reshaped = np.reshape(data_set[i], (7, 5))

    # Plotting the original pattern
    plt.subplot(2, num_patterns, i + 1)
    plt.imshow(original_pattern_reshaped, cmap='gray')
    plt.title(f"Original {i+1}")
    plt.axis('off')

    # Process the reproduced pattern
    discretized_pattern = reproduced_patterns[i].copy()
    discretized_pattern[discretized_pattern > 0.5] = 1
    discretized_pattern[discretized_pattern <= 0.5] = 0
    discretized_pattern = np.reshape(discretized_pattern, (7, 5))

    # Plotting the reproduced pattern
    plt.subplot(2, num_patterns, num_patterns + i + 1)
    plt.imshow(discretized_pattern, cmap='gray')
    plt.title(f"Reproduced {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

