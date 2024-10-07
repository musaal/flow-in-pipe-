# from model import Transformer
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# from helpers import *
# from joblib import load
# from Custom_Loss import custom_loss_function  # Import the custom loss function
# import logging
# from plot import *

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
# logger = logging.getLogger(__name__)

# def train_teacher_forcing(dataloader, EPOCH, frequency, k, path_to_save_model, path_to_save_loss, path_to_save_predictions, pipe_radius, device):
#     # Set device (CPU or GPU)
#     device = torch.device(device)

#     # Initialize the model
#     model = Transformer().double().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adjust learning rate if needed

#     best_model = ""
#     min_train_loss = float('inf')

#     # Load the scaler once outside the loop
#     scaler = load('particle_scaler.joblib')

#     # Training loop
#     for epoch in range(EPOCH + 1):
#         train_loss = 0
#         model.train()

#         for _input, target in dataloader:
#             optimizer.zero_grad()

#             # Limit sequence length to avoid OOM errors
#             max_sequence_length = 1000
#             src = _input.double().to(device)[:, :max_sequence_length, :]  # Limit input length
#             target = target.double().to(device)

#             # Predictions for each step
#             predictions = []
#             num_iterations = 10  # Adjust based on memory constraints

#             for step in range(num_iterations):
#                 predicted_output = model(src, device)
#                 predictions.append(predicted_output)
#                 src = predicted_output  # Use predicted output as next input

#             # Last prediction for loss calculation
#             pred_positions = predictions[-1]
#             pred_velocities = torch.zeros_like(pred_positions)  # Replace with actual velocity calculation

#             # Compute custom loss
#             loss = custom_loss_function(pred_positions, pred_velocities, target, pipe_radius)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional: Gradient clipping
#             optimizer.step()

#             train_loss += loss.detach().item()

#         # Save model with best loss
#         if train_loss < min_train_loss:
#             torch.save(model.state_dict(), f"{path_to_save_model}/best_train_{epoch}.pth")
#             torch.save(optimizer.state_dict(), f"{path_to_save_model}/optimizer_{epoch}.pth")
#             min_train_loss = train_loss
#             best_model = f"best_train_{epoch}.pth"

#         for epoch in range(EPOCH + 1):
#             train_loss = 0
#             model.train()

#             for _input, target in dataloader:
#                 optimizer.zero_grad()

#                 # Limit sequence length to avoid OOM errors
#                 max_sequence_length = 1000
#                 src = _input.double().to(device)[:, :max_sequence_length, :]  # Limit input length
#                 target = target.double().to(device)

#                 # Log the shapes of input and target
#                 logger.info(f"Input shape: {_input.shape}")
#                 logger.info(f"Target shape: {target.shape}")

#                 # Predictions for each step
#                 predictions = []
#                 num_iterations = 10  # Adjust based on memory constraints

#                 for step in range(num_iterations):
#                     predicted_output = model(src, device)
#                     predictions.append(predicted_output)
#                     src = predicted_output  # Use predicted output as next input

#                 # Last prediction for loss calculation
#                 pred_positions = predictions[-1]

#                 # Ensure pred_positions has the same number of dimensions as target
#                 if pred_positions.dim() > 2:
#                     pred_positions = pred_positions[:, -1, :]  # Take only the last output if it's 3D
                
#                 # Assuming pred_positions also has shape [1, 8] after slicing
#                 pred_velocities = torch.zeros_like(pred_positions)  # Replace with actual velocity calculation

#                 # Compute custom loss
#                 loss = custom_loss_function(pred_positions, pred_velocities, target, pipe_radius)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional: Gradient clipping
#                 optimizer.step()

#                 train_loss += loss.detach().item()

#             # Save model with best loss
#             if train_loss < min_train_loss:
#                 torch.save(model.state_dict(), f"{path_to_save_model}/best_train_{epoch}.pth")
#                 torch.save(optimizer.state_dict(), f"{path_to_save_model}/optimizer_{epoch}.pth")
#                 min_train_loss = train_loss
#                 best_model = f"best_train_{epoch}.pth"

#             # Logging every `frequency` epochs
#             if epoch % frequency == 0:
#                 logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")

#                 # Print shapes for debugging
#                 logger.info(f"src shape: {src.shape}")  # Log without slicing
#                 logger.info(f"target shape: {target.shape}")  # Log without slicing
#                 logger.info(f"prediction shape: {pred_positions.shape}")  # Use pred_positions

#                 # Inverse transform using the scaler and detaching tensors
#                 src_humidity = scaler.inverse_transform(src[:, :, 0].detach().cpu().numpy())  # Adjust if needed
#                 target_humidity = scaler.inverse_transform(target.detach().cpu().numpy())  # Use as-is
#                 prediction_humidity = scaler.inverse_transform(pred_positions.detach().cpu().numpy())  # Use as-is

#                 # Plot training results
#                 plot_training(epoch, path_to_save_predictions, src_humidity, prediction_humidity)

#             # Average training loss
#             train_loss /= len(dataloader)
#             log_loss(train_loss, path_to_save_loss, train=True)

#         return best_model

from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from helpers import *
from joblib import load
from Custom_Loss import custom_loss_function  # Import the custom loss function
import logging
from plot import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def train_teacher_forcing(dataloader, EPOCH, frequency, k, path_to_save_model, path_to_save_loss, path_to_save_predictions, pipe_radius, device):
    # Set device (CPU or GPU)
    device = torch.device(device)

    # Initialize the model
    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adjust learning rate if needed

    best_model = ""
    min_train_loss = float('inf')

    # Load the scaler once outside the loop
    scaler = load('particle_scaler.joblib')

    # Training loop
    for epoch in range(EPOCH + 1):
        train_loss = 0
        model.train()

        for _input, target in dataloader:
            optimizer.zero_grad()

            # Limit sequence length to avoid OOM errors
            max_sequence_length = 1000
            src = _input.double().to(device)[:, :max_sequence_length, :]  # Limit input length
            target = target.double().to(device)

            # Predictions for each step
            predictions = []
            num_iterations = 10  # Adjust based on memory constraints

            for step in range(num_iterations):
                predicted_output = model(src, device)
                predictions.append(predicted_output)
                src = predicted_output  # Use predicted output as next input

            # Last prediction for loss calculation
            pred_positions = predictions[-1]
            pred_velocities = torch.zeros_like(pred_positions)  # Replace with actual velocity calculation

            # Compute custom loss
            loss = custom_loss_function(pred_positions, pred_velocities, target, pipe_radius)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional: Gradient clipping
            optimizer.step()

            train_loss += loss.detach().item()

        # Save model with best loss
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), f"{path_to_save_model}/best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(), f"{path_to_save_model}/optimizer_{epoch}.pth")
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}.pth"

        # Logging every `frequency` epochs
        if epoch % frequency == 0:
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")

            # Log the shapes of input and target
            logger.info(f"Input shape: {_input.shape}")
            logger.info(f"Target shape: {target.shape}")

            # Ensure src can be used for inverse transformation
            original_input = _input.double().to(device)  # Keep original input for inverse transform
            src_humidity = scaler.inverse_transform(original_input.detach().cpu().numpy().reshape(-1, original_input.shape[-1]))  # Reshape if necessary

            # Ensure predictions have the correct dimensions for inverse transformation
            if pred_positions.dim() > 2:
                pred_positions = pred_positions[:, -1, :]  # Take only the last output if it's 3D
            prediction_humidity = scaler.inverse_transform(pred_positions.detach().cpu().numpy())

            # Log shapes for debugging
            logger.info(f"src shape: {src.shape}")  # Log without slicing
            logger.info(f"prediction shape: {pred_positions.shape}")  # Use pred_positions

            # Plot training results
            plot_training(epoch, path_to_save_predictions, src_humidity, prediction_humidity)

        # Average training loss
        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)

    return best_model
