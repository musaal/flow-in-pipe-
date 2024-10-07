import os, shutil

# Save training or validation loss to a file
def log_loss(loss_val: float, path_to_save_loss: str, train: bool = True):
    file_name = "train_loss.txt" if train else "val_loss.txt"

    # Ensure the directory exists
    path_to_file = os.path.join(path_to_save_loss, file_name)
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    # Log the loss
    with open(path_to_file, "a") as f:
        f.write(f"{loss_val}\n")  # Added newline for clarity


# Exponential Moving Average, https://en.wikipedia.org/wiki/Moving_average
def EMA(values, alpha=0.1):
    if not values:  # Handle empty list
        return []
    ema_values = [values[0]]
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha * item + (1 - alpha) * ema_values[idx])
    return ema_values


# Clean directories from previous runs and recreate the necessary folders
def clean_directory():
    dirs_to_clean = ['save_loss', 'save_model', 'save_predictions']
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
