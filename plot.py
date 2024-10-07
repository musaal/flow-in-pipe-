# import matplotlib.pyplot as plt
# from helpers import EMA
# import numpy as np
# import torch

# def plot_loss(path_to_save, train=True):
#     """Plot and save training or validation loss."""
#     plt.rcParams.update({'font.size': 10})
#     with open(path_to_save + "/train_loss.txt", 'r') as f:
#         loss_list = [float(line) for line in f.readlines()]
    
#     title = "Train" if train else "Validation"
#     EMA_loss = EMA(loss_list)
    
#     plt.figure(figsize=(10, 5))  # Added figure size for better visualization
#     plt.plot(loss_list, label="Loss")
#     plt.plot(EMA_loss, label="EMA Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.title(f"{title} Loss")
#     plt.savefig(path_to_save + f"/{title}.png")
#     plt.close()

# def plot_prediction(title, path_to_save, src, tgt, prediction, dims=6):
#     """Plot and save multi-dimensional predictions against source and target."""
    
#     # Create a subplot for each dimension
#     fig, axes = plt.subplots(dims, 1, figsize=(15, 6 * dims))  # Adjusting figure size based on dimensions
#     plt.rcParams.update({"font.size": 16})

#     # Plotting the input, target, and predicted values for each dimension
#     for dim in range(dims):
#         idx_scr = [i for i in range(len(src[:, dim]))]  # Index for the source data
#         idx_tgt = [i for i in range(len(tgt[:, dim]))]  # Index for the target data
#         idx_pred = [i for i in range(1, len(prediction[:, dim]) + 1)]  # Index for predicted data

#         ax = axes[dim]  # Selecting the correct subplot axis
#         ax.plot(idx_scr, src[:, dim], '-', color='blue', label=f'Input Flow Data (Dim {dim+1})', linewidth=2)
#         ax.plot(idx_tgt, tgt[:, dim], '-', color='indigo', label=f'Target Flow Data (Dim {dim+1})', linewidth=2)
#         ax.plot(idx_pred, prediction[:, dim], '--', color='limegreen', label=f'Forecast Flow Data (Dim {dim+1})', linewidth=2)

#         # Formatting for each dimension plot
#         ax.grid(b=True, which='major', linestyle='solid')
#         ax.minorticks_on()
#         ax.grid(b=True, which='minor', linestyle='dashed', alpha=0.5)
#         ax.set_xlabel("Time Elapsed")
#         ax.set_ylabel("Flow Rate (units)")
#         ax.legend()
#         ax.set_title(f"Dimension {dim + 1} - Forecast Flow Data - {title}")

#     # Save plot
#     plt.tight_layout()
#     plt.savefig(path_to_save + f"/Prediction_{title}.png")
#     plt.close()


#     # Save plot
#     plt.savefig(path_to_save + f"/Prediction_{title}.png")
#     plt.close()

# def plot_training(epoch, path_to_save, src, prediction, dims=6):
#     """Plot and save multi-dimensional training results for a specific epoch."""
    
#     fig, axes = plt.subplots(dims, 1, figsize=(15, 6 * dims))  # Adjusting figure size based on dimensions
#     plt.rcParams.update({"font.size": 18})
    
#     for dim in range(dims):
#         idx_scr = [i for i in range(len(src[:, dim]))]  # Index for the source data
#         idx_pred = [i for i in range(1, len(prediction[:, dim]) + 1)]  # Index for predicted data

#         ax = axes[dim]  # Selecting the correct subplot axis
#         ax.plot(idx_scr, src[:, dim], 'o-.', color='blue', label=f'Input Flow Data (Dim {dim+1})', linewidth=1)
#         ax.plot(idx_pred, prediction[:, dim], 'o-.', color='limegreen', label=f'Prediction Flow Data (Dim {dim+1})', linewidth=1)

#         ax.grid(visible=True, which='major', linestyle='-')
#         ax.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
#         ax.minorticks_on()
#         ax.set_xlabel("Time Elapsed")
#         ax.set_ylabel("Flow Rate (units)")
#         ax.legend()
#         ax.set_title(f"Training Epoch {epoch} - Dimension {dim + 1}")

#     # Save plot
#     plt.tight_layout()
#     plt.savefig(path_to_save + f"/Epoch_{str(epoch)}.png")
#     plt.close()


# def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, dims=6):
#     """Plot and save multi-dimensional training results with sampled sources for a specific epoch."""
    
#     fig, axes = plt.subplots(dims, 1, figsize=(15, 6 * dims))  # Adjusting figure size based on dimensions
#     plt.rcParams.update({"font.size": 18})
    
#     for dim in range(dims):
#         idx_scr = [i for i in range(len(src[:, dim]))]  # Index for the source data
#         idx_sampled_src = [i for i in range(len(sampled_src[:, dim]))]  # Index for sampled source data
#         idx_pred = [i for i in range(1, len(prediction[:, dim]) + 1)]  # Index for predicted data

#         ax = axes[dim]  # Selecting the correct subplot axis
#         ax.plot(idx_sampled_src, sampled_src[:, dim], 'o-.', color='red', label=f'Sampled Flow Data (Dim {dim+1})', linewidth=1, markersize=10)
#         ax.plot(idx_scr, src[:, dim], 'o-.', color='blue', label=f'Input Flow Data (Dim {dim+1})', linewidth=1)
#         ax.plot(idx_pred, prediction[:, dim], 'o-.', color='limegreen', label=f'Prediction Flow Data (Dim {dim+1})', linewidth=1)

#         ax.grid(b=True, which='major', linestyle='-')
#         ax.grid(b=True, which='minor', linestyle='--', alpha=0.5)
#         ax.minorticks_on()
#         ax.set_xlabel("Time Elapsed")
#         ax.set_ylabel("Flow Rate (units)")
#         ax.legend()
#         ax.set_title(f"Training Epoch {epoch} - Dimension {dim + 1}")

#     # Save plot
#     plt.tight_layout()
#     plt.savefig(path_to_save + f"/Epoch_{str(epoch)}.png")
#     plt.close()

import matplotlib.pyplot as plt
from helpers import EMA
import numpy as np
import torch
import logging
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to create a consistent plotting style
def set_plot_style():
    plt.rcParams.update({'font.size': 16})  # Set font size without specific style

# Helper to format axes, legends, and grids
def format_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', linestyle='--', alpha=0.5)
    ax.minorticks_on()

# Function to plot EMA and Loss
def plot_loss(path_to_save, train=True):
    """Plot and save training or validation loss."""
    try:
        set_plot_style()
        title = "Train" if train else "Validation"
        with open(path_to_save + "/train_loss.txt", 'r') as f:
            loss_list = [float(line.strip()) for line in f.readlines()]

        EMA_loss = EMA(loss_list)

        plt.figure(figsize=(12, 6))
        plt.plot(loss_list, label="Loss", color='tab:blue')
        plt.plot(EMA_loss, label="EMA Loss", color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{title} Loss Over Epochs")
        plt.legend()

        plt.savefig(f"{path_to_save}/{title}_Loss.png", format='png')
        plt.close()
        logging.info(f"{title} loss plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_loss: {str(e)}")

# Function to plot predictions for multi-dimensional data
def plot_prediction(title, path_to_save, src, tgt, prediction, dims=6):
    """Plot and save multi-dimensional predictions against source and target."""
    try:
        set_plot_style()
        fig, axes = plt.subplots(dims, 1, figsize=(15, 6 * dims))
        sns.color_palette("husl", 8)

        for dim in range(dims):
            idx_scr = np.arange(len(src[:, dim]))  # Index for source
            idx_tgt = np.arange(len(tgt[:, dim]))  # Index for target
            idx_pred = np.arange(1, len(prediction[:, dim]) + 1)  # Index for predicted

            ax = axes[dim]
            ax.plot(idx_scr, src[:, dim], '-', color='blue', label=f'Input (Dim {dim+1})')
            ax.plot(idx_tgt, tgt[:, dim], '-', color='indigo', label=f'Target (Dim {dim+1})')
            ax.plot(idx_pred, prediction[:, dim], '--', color='limegreen', label=f'Forecast (Dim {dim+1})')

            format_axes(ax, "Time Elapsed", "Flow Rate (units)", f"Dimension {dim + 1} - {title}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{path_to_save}/Prediction_{title}.png", format='png')
        plt.close()
        logging.info(f"Prediction plot for {title} saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_prediction: {str(e)}")

# Optimized plotting for training with parallelization
def plot_training(epoch, path_to_save, src, prediction, dims=6):
    """Parallelized plot for multi-dimensional training results."""
    try:
        set_plot_style()
        fig, axes = plt.subplots(dims, 1, figsize=(15, 6 * dims))
        sns.color_palette("Set1", 8)

        def plot_dimension(dim):
            idx_scr = np.arange(len(src[:, dim]))
            idx_pred = np.arange(1, len(prediction[:, dim]) + 1)

            ax = axes[dim]
            ax.plot(idx_scr, src[:, dim], 'o-.', color='blue', label=f'Input (Dim {dim+1})')
            ax.plot(idx_pred, prediction[:, dim], 'o-.', color='limegreen', label=f'Prediction (Dim {dim+1})')

            format_axes(ax, "Time Elapsed", "Flow Rate (units)", f"Epoch {epoch} - Dim {dim + 1}")
            ax.legend()

        # Use multithreading to speed up plotting for large dimensions
        with ThreadPoolExecutor(max_workers=dims) as executor:
            executor.map(plot_dimension, range(dims))

        plt.tight_layout()
        plt.savefig(f"{path_to_save}/Epoch_{str(epoch)}.png", format='png')
        plt.close()
        logging.info(f"Training plot for Epoch {epoch} saved successfully.")
    except Exception as e:
        logging.error(f"Error in parallel_plot_training: {str(e)}")

# Plot training with sampled data
def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, dims=6):
    """Plot and save training results with sampled sources for a specific epoch."""
    try:
        set_plot_style()
        fig, axes = plt.subplots(dims, 1, figsize=(15, 6 * dims))
        sns.color_palette("coolwarm", dims)

        for dim in range(dims):
            idx_scr = np.arange(len(src[:, dim]))
            idx_sampled_src = np.arange(len(sampled_src[:, dim]))
            idx_pred = np.arange(1, len(prediction[:, dim]) + 1)

            ax = axes[dim]
            ax.plot(idx_sampled_src, sampled_src[:, dim], 'o-.', color='red', label=f'Sampled (Dim {dim+1})')
            ax.plot(idx_scr, src[:, dim], 'o-.', color='blue', label=f'Input (Dim {dim+1})')
            ax.plot(idx_pred, prediction[:, dim], 'o-.', color='limegreen', label=f'Prediction (Dim {dim+1})')

            format_axes(ax, "Time Elapsed", "Flow Rate (units)", f"Epoch {epoch} - Dim {dim + 1}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{path_to_save}/Epoch_{str(epoch)}_Sampled.png", format='png')
        plt.close()
        logging.info(f"Sampled training plot for Epoch {epoch} saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_training_with_sample: {str(e)}")

