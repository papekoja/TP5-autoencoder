import dill

def save_model_and_config(model, lowest_error, config, save_path):
    with open(save_path, 'wb') as file_handle:
        dill.dump(model, file_handle)

    config_contents = str(config)  # Convert config dict to string

    text_filename = save_path.replace('.pkl', '.txt')  # Replace the extension

    with open(text_filename, 'w') as file_handle:
        file_handle.write("Configuration:\n")
        file_handle.write(config_contents)
        file_handle.write("\n\nLowest Error Achieved:\n")
        file_handle.write(str(lowest_error))
