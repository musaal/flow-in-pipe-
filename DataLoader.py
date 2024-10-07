import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
from joblib import dump
import logging 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

class PipeFlowDataset(Dataset):
    def __init__(self, particle_data_file, flow_data_file, root_dir, training_length, forecast_window):
        """
        Args:
            particle_data_file (string): Path to the CSV file with particle data (x, y, z, Vx, Vy, Vz).
            flow_data_file (string): Path to the CSV file with pipe flow parameters (y^+, U_z^+, P^+, etc.).
            root_dir (string): Directory containing the CSV files.
            training_length (int): Length of the training sequence for each batch.
            forecast_window (int): Length of the forecast window.
        """
        self.particle_file = os.path.join(root_dir, particle_data_file)
        self.flow_file = os.path.join(root_dir, flow_data_file)

        # Load particle data and flow data
        self.particle_df = pd.read_csv(self.particle_file)
        self.flow_df = pd.read_csv(self.flow_file)

        # Initialize scalers for normalizing
        self.particle_scaler = MinMaxScaler()
        self.flow_scaler = MinMaxScaler()

        # Normalize particle data
        particle_columns = ["x", "y", "z", "vx", "vy", "vz"]
        self.particle_df[particle_columns] = self.particle_scaler.fit_transform(self.particle_df[particle_columns])

        # Normalize flow data
        flow_columns = ['y^+', 'U_z^+', 'P^+', 'u_z^2', 'u_r^2', 'u_t^2', 'UzUr', 'p^2']
        self.flow_df[flow_columns] = self.flow_scaler.fit_transform(self.flow_df[flow_columns])

        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # Number of 500,000 particle chunks available in the dataset
        return len(self.particle_df) // 500000

    def __getitem__(self, idx):
        """
        This method retrieves the particle data and the corresponding flow data for each index.
        """
        # Get the range for this batch
        start = idx * 500000
        end = start + 500000

        # Get input data (particles' position and velocity)
        particle_data = self.particle_df.iloc[start:end][["x", "y", "z", "vx", "vy", "vz"]].values
        particle_data = torch.tensor(particle_data, dtype=torch.float32)

        # Get corresponding flow parameters for this batch as targets
        target_flow = self.flow_df.iloc[idx].values  # Single row corresponding to the whole batch
        target_flow = torch.tensor(target_flow, dtype=torch.float32)

        dump(self.particle_scaler, 'particle_scaler.joblib')
        dump(self.flow_scaler, 'flow_scaler.joblib')   

        return particle_data, target_flow

    # def save_scalers(self):
    #     dump(self.particle_scaler, 'particle_scaler.joblib')
    #     dump(self.flow_scaler, 'flow_scaler.joblib')                                                     