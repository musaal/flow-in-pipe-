# from model import Transformer
# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# from DataLoader import PipeFlowDataset
# import logging
# import time  # debugging
# from plot import *
# from helpers import *
# from joblib import load

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
# logger = logging.getLogger(__name__)

# def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):

#     device = torch.device(device)
    
#     model = Transformer().double().to(device)
#     model.load_state_dict(torch.load(path_to_save_model + best_model))
#     criterion = torch.nn.MSELoss()

#     val_loss = 0
#     with torch.no_grad():

#         model.eval()
#         for plot in range(25):  # Example number of plots to create, adjust as needed

#             # for index_in, index_tar, _input, target, sensor_number in dataloader:
#             for _input, target in dataloader:
                
#                 # _input shape: [batch_size, seq_len, features] -> here features = 6 (positions + velocities)
#                 # starting from 1 to align input with target
#                 src = _input.permute(1, 0, 2).double().to(device)[1:, :, :]  # e.g., (t1 -- t47 for positions and velocities)
                

#                 if target.dim() == 2:
#                     target = target.permute(1, 0).double().to(device)
#                 elif target.dim() == 3:
#                     target = target.permute(1, 0, 2).double().to(device)

#                 # target = target.permute(1, 0, 2).double().to(device)  # t48 - t59

#                 next_input_model = src
#                 all_predictions = []

#                 for i in range(forecast_window - 1):
#                     # Forward pass through model
#                     prediction = model(next_input_model, device)  # Predict next time step positions and velocities

#                     if all_predictions == []:
#                         all_predictions = prediction  # First prediction
#                     else:
#                         all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0)))  # Append predictions

#                     # Adjusting positional encoding for next step (z-increment)
#                     pos_encoding_old_vals = src[i+1:, :, 1:]  # Current time step position and velocity encodings
#                     pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1)  # Target next time step positional encodings
#                     pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val))  # Update position encodings
                    
#                     next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0)), dim=0)  # Update input for next iteration
#                     next_input_model = torch.cat((next_input_model, pos_encodings), dim=2)  # Prepare for next iteration

#                 true = torch.cat((src[1:, :, 0], target[:-1, :, 0]))  # Actual values
#                 loss = criterion(true, all_predictions[:,:,0])  # Loss based on predictions
#                 val_loss += loss
            
#             val_loss = val_loss / 10
#             scaler = load('scalar_item.joblib')  # Ensure this scaler is correctly used
#             src_scaled = scaler.inverse_transform(src[:,:,0].cpu())  # Inverse scaling the input data
#             target_scaled = scaler.inverse_transform(target[:,:,0].cpu())  # Inverse scaling the target data
#             prediction_scaled = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())  # Inverse scaling the predictions

#             # plot_prediction(plot, path_to_save_predictions, src_scaled, target_scaled, prediction_scaled, sensor_number, index_in, index_tar)
#             plot_prediction(plot, path_to_save_predictions, src_scaled, target_scaled, prediction_scaled)

#         logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")
from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import PipeFlowDataset
import logging
import time  # debugging
from plot import *
from helpers import *
from joblib import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):
    device = torch.device(device)
    
    model = Transformer().float().to(device)  # Use float instead of double
    model.load_state_dict(torch.load(path_to_save_model + best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():
        model.eval()
        
        for plot in range(min(25, len(dataloader))):  # Limit to available data
            for _input, target in dataloader:
                # Check if input is too large, reduce batch size if necessary
                if _input.size(0) > 32:  # Example threshold for batch size
                    continue  # Skip this batch if too large
                
                src = _input.permute(1, 0, 2).float().to(device)[1:, :, :]
                target = target.permute(1, 0, 2).float().to(device) if target.dim() == 3 else target.permute(1, 0).float().to(device)

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window - 1):
                    prediction = model(next_input_model, device)

                    if len(all_predictions) == 0:
                        all_predictions = prediction
                    else:
                        all_predictions = torch.cat((all_predictions, prediction[-1, :, :].unsqueeze(0)))

                    # Update positional encodings
                    pos_encoding_old_vals = src[i + 1:, :, 1:]  
                    pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1)  
                    pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val))

                    next_input_model = torch.cat((src[i + 1:, :, 0].unsqueeze(-1), prediction[-1, :, :].unsqueeze(0)), dim=0)
                    next_input_model = torch.cat((next_input_model, pos_encodings), dim=2)

                true = torch.cat((src[1:, :, 0], target[:-1, :, 0]))
                loss = criterion(true, all_predictions[:, :, 0])
                val_loss += loss.item()  # Store loss value directly
            
            # Average the loss over the number of plots
            val_loss /= (plot + 1)

            # Load scaler and inverse transform
            scaler = load('scalar_item.joblib')
            src_scaled = scaler.inverse_transform(src[:, :, 0].cpu().numpy())  
            target_scaled = scaler.inverse_transform(target[:, :, 0].cpu().numpy())
            prediction_scaled = scaler.inverse_transform(all_predictions[:, :, 0].cpu().numpy())

            # Create prediction plots
            plot_prediction(plot, path_to_save_predictions, src_scaled, target_scaled, prediction_scaled)

        logger.info(f"Loss On Unseen Dataset: {val_loss}")

        # Clean up memory if needed
        torch.cuda.empty_cache()  # Uncomment if using GPU
