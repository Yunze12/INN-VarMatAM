"""
Evaluation script for the INN model's forward prediction capabilities.

This script loads a trained INN model, evaluates its performance on random batches of data,
and calculates key metrics including RMSE and R² scores for both forward and backward predictions.

It also provides sample input and prediction outputs for better understanding of the model's behavior.
"""

from train_config import *
from data_processing import load_and_preprocess_data, create_dataset, inverse_minmax_normal
from model import INN
import numpy as np
from sklearn.metrics import r2_score
import torch
import random
from torch.utils.data import DataLoader, TensorDataset


def evaluate_forward_prediction(model_path, test_data_path):
    """
    Evaluate the forward prediction capabilities of the INN model on the test set.

    Args:
        model_path (str): Path to the trained model checkpoint.
        test_data_path (str): Path to the test dataset.
    """
    # Storage for collected predictions and targets
    all_predictions = []
    all_targets = []
    all_x_pre = []

    try:
        # Load trained model
        model = INN().to(device)
        checkpoint = torch.load(eval_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load test dataset
        test_data = torch.load(test_data_path, weights_only=True)
        test_x = test_data['x']
        test_y = test_data['y']
        min_x = test_data['min_x']
        max_x = test_data['max_x']
        min_y = test_data['min_y']
        max_y = test_data['max_y']

        # Create test dataset and dataloader
        test_dataset = TensorDataset(test_x, test_y)
        test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        # Evaluate on the test set
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)

                # Make predictions
                predictions = model.forward(inputs)  # Forward prediction: x -> y
                reconstructed_x = model.inverse(targets)  # Backward prediction: y -> x

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets[:, 0].cpu().numpy())  # Assuming single output dimension
                all_x_pre.extend(reconstructed_x.cpu().numpy())

        # Convert to numpy arrays for easier manipulation
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_x_pre = np.array(all_x_pre)

        # Calculate forward prediction metrics
        rmse_forward = np.sqrt(np.mean((all_targets - all_predictions[:, 0]) ** 2))
        r2_forward = r2_score(all_targets, all_predictions[:, 0])
        print(f"\nForward Prediction Metrics:")
        print(f"RMSE: {rmse_forward:.3f}")
        print(f"R²: {r2_forward:.3f}")

        # Denormalize data for backward prediction metrics
        original_x = inverse_minmax_normal(test_x.cpu(), min_x.cpu(), max_x.cpu()).numpy()
        reconstructed_x_denorm = inverse_minmax_normal(torch.from_numpy(all_x_pre), min_x.cpu(), max_x.cpu()).numpy()

        # Calculate backward prediction metrics
        rmse_backward = np.sqrt(np.mean((original_x - reconstructed_x_denorm) ** 2))
        r2_backward = r2_score(original_x, reconstructed_x_denorm)
        print(f"\nInverse Prediction Metrics(without optimization):")
        print(f"RMSE: {rmse_backward:.3f}")
        print(f"R²: {r2_backward:.3f}")

        # Display sample results
        print("\nSample Input and Predictions:")
        for i in range(min(1, len(test_x))):
            print(f"Sample {i + 1}:")
            print(f"  Input x: {original_x[i]}")
            print(f"  Reconstructed x: {reconstructed_x_denorm[i]}")
            print(f"  Target y: {all_targets[i]}")
            print(f"  Predicted y: {all_predictions[i, 0]}")
            print(f"  Reconstruction Error: {np.linalg.norm(original_x[i] - reconstructed_x_denorm[i]):.3f}")
            print(f"  Prediction Error: {abs(all_targets[i] - all_predictions[i, 0]):.3f}\n")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


if __name__ == "__main__":

    try:
        # Run evaluation
        evaluate_forward_prediction(model_path, test_data_path)

    except Exception as e:
        print(f"Error during evaluation setup: {str(e)}")
