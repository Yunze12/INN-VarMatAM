import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from train_config import *


def load_and_preprocess_data(data_file_path):
    """
    Load data from the specified Excel file and perform preprocessing.

    Args:
        data_file_path (str): The path to the Excel data file.

    Returns:
        tuple: Preprocessed input data, target data, and normalization parameters.
    """
    try:
        # Load data
        data = pd.read_excel(data_file_path)

        # Extract inputs and outputs
        inputs = data.iloc[:, :3]  # First three columns as inputs
        outputs = data.iloc[:, 3]  # Fourth column as output

        # Convert to tensors
        x = torch.tensor(inputs.to_numpy(dtype=np.float32))
        y = torch.tensor(outputs.to_numpy(dtype=np.float32)).unsqueeze(1)

        # Compute normalization parameters
        min_x = x.min(dim=0).values
        max_x = x.max(dim=0).values
        min_y = y.min(dim=0).values
        max_y = y.max(dim=0).values

        # Normalize data
        normal_x = minmax_normal(x, min_x, max_x)
        normal_y = minmax_normal(y, min_y, max_y)

        # Add random tensor to y
        torch.manual_seed(seed)
        random_tensor = torch.randn(len(outputs), ndim_z)
        normal_y = torch.cat((normal_y, random_tensor), dim=1)

        return normal_x, normal_y, min_x, max_x, min_y, max_y

    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise


def create_dataset(x, y):
    """
    Create a custom dataset.

    This function creates a custom PyTorch dataset based on the given input and target data.

    Args:
        x (torch.Tensor): Input data tensor.
        y (torch.Tensor): Target data tensor.

    Returns:
        torch.utils.data.Dataset: A custom dataset object.
    """
    class MyDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index], self.y[index]

    return MyDataset(x, y)


def minmax_normal(tensor, min_val, max_val):
    """
    Min-max normalization for tensors.

    Args:
        tensor (torch.Tensor): Input tensor.
        min_val (torch.Tensor): Minimum values for normalization.
        max_val (torch.Tensor): Maximum values for normalization.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    return (tensor - min_val) / (max_val - min_val)


def inverse_minmax_normal(tensor, min_val, max_val):
    """
    Inverse min-max normalization to revert tensors to original scale.

    Args:
        tensor (torch.Tensor): Normalized tensor.
        min_val (torch.Tensor): Minimum values used for normalization.
        max_val (torch.Tensor): Maximum values used for normalization.

    Returns:
        torch.Tensor: Denormalized tensor.
    """
    return tensor * (max_val - min_val) + min_val