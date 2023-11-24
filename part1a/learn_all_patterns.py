import numpy as np
import matplotlib.pyplot as plt
import utils.file_readers as file_readers
import utils.activation_functions as activation_functions
import models.autoencoder as autoencoder

# Read the font data
config: dict = file_readers.read_config('config.json')

# Read the font data
font_data = file_readers.read_fonts_h(config['data_path'])
font_data = np.array(font_data)

# Initializes the auto-encoder
activation_function, derived_activation_function = activation_functions.get_activation_functions(
    config["activation_function"], config["beta"])
auto_encoder = autoencoder.AutoEncoder(activation_function, derived_activation_function, len(font_data[0]),
                                       config["hidden_layout"], config["latent_dim"],
                                       config["momentum"], config["momentum_alpha"])

# Use all patterns for training
data_set = np.array(font_data)

# Train auto-encoder
for epoch in range(config["epochs"]):
    for data in data_set:
        auto_encoder.train(data, data, config["learning_rate"])
    auto_encoder.update_weights()
    error = auto_encoder.compute_error(data_set, data_set)
    if error < config["error_threshold"]:
        break
    if epoch % 50 == 0:
        print(f'Iteration {epoch}, error {error}')

# Reproduce all patterns
reproduced_patterns = auto_encoder.forward_pass(data_set)

# Process and plot each pattern
num_patterns = len(data_set)
for i in range(num_patterns):
    # Convert values: 1 if above 0.5, 0 otherwise
    discretized_pattern = reproduced_patterns[i].copy()
    discretized_pattern[discretized_pattern > 0.5] = 1
    discretized_pattern[discretized_pattern <= 0.5] = 0
    discretized_pattern = np.reshape(discretized_pattern, (7, 5))
    original_pattern_reshaped = np.reshape(data_set[i], (7, 5))

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Pattern {i+1}")
    plt.imshow(original_pattern_reshaped, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Reproduced Pattern {i+1}")
    plt.imshow(discretized_pattern, cmap='gray')
    plt.axis('off')

    plt.show()
