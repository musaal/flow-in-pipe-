# i will give you all the code we are working with so far make sure all the code follow the same structure,
# and our goal to use to data (x,y,z,Vx,Vy,Vz) containing the 500000 particles and the other data is for
# the flow and have the y^+,U_z^+,P^+,u_z^2,u_r^2,u_t^2,UzUr,p^2  and 910  row of data, we try to make the model prodicat
# the next 500000 particles and transform this prediction into comparable value with the folw data and this the all codes  
# so fare check for any issue with the codes :

import argparse
from train_teacher_forcing import *
from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *

def main(
    epoch: int = 100,
    pipe_radius = 1.0 ,
    k: int = 60,
    batch_size: int = 1,
    frequency: int = 10,
    training_length = 48,
    forecast_window = 24,
    train_csv = "pipe_flow_slice_0.1.csv", 
    test_csv = "pipe_flow_slice_test.csv",
    path_to_save_model = "save_model/",
    path_to_save_loss = "save_loss/", 
    path_to_save_predictions = "save_predictions/", 
    device = "cpu" #"cpu"
):

    clean_directory()

    # Adjust data loading to fit the input and target shapes
    train_dataset = PipeFlowDataset(particle_data_file=train_csv,flow_data_file = "flowp.csv" , root_dir="Data/", training_length=training_length, forecast_window=forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # After initializing your PipeFlowDataset, call this method


    test_dataset = PipeFlowDataset(particle_data_file=test_csv,flow_data_file = "flowp.csv" , root_dir="Data/", training_length=training_length, forecast_window=forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Example of inspecting DataLoader structure
    for batch_index, (inputs, targets) in enumerate(train_dataloader):
        print(f"Batch {batch_index + 1}:")
        print(f"Inputs shape: {inputs.shape}")  # Shape of the input data
        print(f"Targets shape: {targets.shape}")  # Shape of the target data
        if batch_index == 2:  # Print only the first 3 batches to avoid clutter
            break

    # Train and save the best model
    best_model = train_teacher_forcing(train_dataloader, epoch , frequency, k, path_to_save_model, path_to_save_loss, path_to_save_predictions, pipe_radius, device)
    # Run inference with the trained model
    # inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=10)
    parser.add_argument("--path_to_save_model", type=str, default="save_model/")
    parser.add_argument("--path_to_save_loss", type=str, default="save_loss/")
    parser.add_argument("--path_to_save_predictions", type=str, default="save_predictions/")
    parser.add_argument("--pipe_radius", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        k=args.k,
        batch_size=args.batch_size,
        frequency=args.frequency,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,
        pipe_radius = args.pipe_radius,
        device=args.device,
    )
