import os
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


#==============================================================================#


@dataclass
class VeinsDatasetConfig:
    csv_paths: List[str]
    column_headers: List[str]
    timesteps: int
    seed: int = field(default=42)


class VeinsDataset():
    def __init__(self, config: VeinsDatasetConfig):
        self.config = config
        self.csv_paths = self.config.csv_paths
        self.column_headers = self.config.column_headers
        self.timesteps = self.config.timesteps
        self.seed = self.config.seed
        self.split = self._make_split()
    
    def _process_dataframe(self):
        """
        Loads, preprocesses, and prepares data for LSTM from a csv file.
        Args:
            csv_paths (str): Path to the CSV file.
            timesteps (int): Number of timesteps for the LSTM.
            test_size (float): Proportion of data for testing.
            validation_size (float): Proportion of data for validation (relative to training).

        Returns:
            tuple: train_loader, val_loader, test_loader
        """
        data_df = pd.DataFrame()
        for csv_path in self.csv_paths:
            df = pd.read_csv(csv_path, names=self.column_headers)
            df = df.drop('Entry', axis=1)
            df = df.drop('Entity', axis=1)
            data_df = pd.concat([data_df, df], ignore_index=True)

        # Extract data and labels
        x = data_df.drop('Collision', axis=1).values # Features
        y = data_df['Collision'].values # Labels
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x = scaler.fit_transform(x)
        return x, y
    
    def _make_split(self, validation_size=0.1, test_size=0.1) -> dict:
        """
        Split into training, test, and validation sets
        """
        x, y = self._process_dataframe()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, shuffle=False, random_state=self.seed
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_size, shuffle=False, random_state=self.seed
        )
        return {
            "train": {"X": x_train, "Y": y_train},
            "val": {"X": x_val, "Y": y_val},
            "test": {"X": x_test, "Y": y_test}
        }
    
    def _create_sequence_data(self, x):
        x_seq = []
        time_steps = self.timesteps
        for i in range(len(x) - time_steps):
            x_seq.append(x[i : (i + time_steps)])
        return np.array(x_seq)
    
    def _create_sequence_data(self, x, y):
        time_steps = self.timesteps
        xs, ys = [], []
        for i in range(len(x) - time_steps):
            xs.append(x[i : (i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(xs), np.array(ys)
    
    def _get_torch_data(self, x, y):
        x, y = self._create_sequence_data(x, y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
        
    def _get_dataset(self, mode: str) -> dict:
        split_dict = self.split.get(mode)
        split_dict["X"], split_dict["Y"] = self._get_torch_data(
            split_dict["X"], split_dict["Y"]
        )
        split_dict["Y"] = split_dict["Y"].view(-1, 1)
        dataset = torch.utils.data.TensorDataset(split_dict["X"], split_dict["Y"])
        return dataset
