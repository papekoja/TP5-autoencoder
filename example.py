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

error_threshold: float = config["error_threshold"]

# activation function and its derived
activation_function, derived_activation_function = activation_functions.get_activation_functions(
    config["activation_function"], config["beta"])

# initializes the auto-encoder
auto_encoder = autoencoder.AutoEncoder(activation_function, derived_activation_function, len(font_data[0]),
                                       config["hidden_layout"], config["latent_dim"],
                                       config["momentum"], config["momentum_alpha"])

data_set = np.array(font_data[:1])

# train auto-encoder
for epoch in range(config["epochs"]):

    # train for this epoch
    for data in data_set:
        auto_encoder.train(data, data, config["learning_rate"])

    # apply the changes
    auto_encoder.update_weights()

    # calculate error
    error = auto_encoder.compute_error(data_set, data_set)

    if error < config["error_threshold"]:
        break

    if epoch % 50 == 0:
        print(f'Iteration {epoch}, error {error}')

reproduced_pattern = auto_encoder.forward_pass(data_set)

# Convert values: 1 if above 0.5, 0 otherwise
discretized_pattern = reproduced_pattern.copy()
discretized_pattern[discretized_pattern > 0.5] = 1
discretized_pattern[discretized_pattern <= 0.5] = 0

# Ensure the discretized pattern is in the correct shape
discretized_pattern = np.reshape(discretized_pattern, (7, 5))

# Original pattern reshaped
original_pattern_reshaped = np.reshape(data_set[0], (7, 5))

# Plotting the original pattern
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.title("Original Pattern")
plt.imshow(original_pattern_reshaped, cmap='gray')
plt.axis('off')

# Plotting the reproduced (discretized) pattern
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.title("Reproduced Pattern")
plt.imshow(discretized_pattern, cmap='gray')
plt.axis('off')

plt.show()
