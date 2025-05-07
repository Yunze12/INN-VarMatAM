"""
Optimization script for the INN model.

This script demonstrates how to use the trained INN model for inverse problems,
specifically generating posterior samples and performing gradient-based optimization
to find the most probable input parameters given a target output.

It also evaluates the model's performance using R² scores and visualizes the posterior distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time
from train_config import *
from data_processing import inverse_minmax_normal
from model import INN
from sklearn.metrics import r2_score
import os

# Ensure directories exist
os.makedirs(figure_path, exist_ok=True)


def load_trained_model(model_path):
    """
    Load a trained INN model from a checkpoint file.

    Args:
        model_path (str): Path to the model checkpoint file.

    Returns:
        INN: Loaded INN model.
    """
    model = INN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def sample_posterior(y, N, model):
    """
    Generate posterior samples from the model given a target output y.

    Args:
        y (torch.Tensor): Target output value.
        N (int): Number of samples to generate.
        model (INN): Trained INN model.

    Returns:
        torch.Tensor: Generated posterior samples.
    """
    rev_inputs = torch.cat([y.expand(N, 1), torch.randn(N, ndim_z).to(device)], dim=1)
    x_samples = model.inverse(rev_inputs)
    return x_samples


def perform_localization(y_target, x_true, model, lr=0.01, n_iter=1000):
    """
    Perform localization (gradient-based optimization) to find the optimal input parameters.

    Args:
        y_target (torch.Tensor): Target output value.
        x_true (torch.Tensor): True input parameters.
        model (INN): Trained INN model.
        lr (float, optional): Learning rate. Defaults to 0.01.
        n_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        torch.Tensor: Optimized input parameters.
    """
    # Generate posterior samples
    x_samples = sample_posterior(y_target, 5000, model)

    # Identify effective samples with a specific target y and delta
    delta = 1.0
    eff_list = []
    out_y1 = model.forward(x_samples)
    rev_x1_eff = x_samples.cpu().data.numpy()
    for j in range(len(out_y1[:, 0])):
        if (y_target + delta > out_y1[j, 0] > y_target - delta
                and x_true[0] + delta > rev_x1_eff[j, 0] > x_true[0] - delta
                and x_true[-1] + delta > rev_x1_eff[j, -1] > x_true[-1] - delta):
            eff_list.append(j)

    # Filter effective samples
    if eff_list:
        x_samples = x_samples[eff_list, :]
        print(f"Effective sample size: {len(eff_list)}")
    else:
        print("No effective samples found. Using all samples for optimization.")

    # Ensure x_samples is a leaf node
    x_samples = x_samples.detach().requires_grad_(True)

    # Perform gradient-based optimization
    x_samples.requires_grad_(True)
    optimizer = torch.optim.Adam([x_samples], lr=lr)

    for k in range(n_iter):
        optimizer.zero_grad()
        out_y1 = model(x_samples)
        re_loss = torch.mean((out_y1[:, 0] - y_target) ** 2)
        re_loss.backward()
        optimizer.step()

    return x_samples.detach()


def evaluate_localization(model, x_all, y_all, min_x, max_x):
    """
    Evaluate the localization performance of the model.

    Args:
        model (INN): Trained INN model.
        x_all (torch.Tensor): All input parameters.
        y_all (torch.Tensor): All output values.
        min_x (torch.Tensor): Minimum values for input data normalization.
        max_x (torch.Tensor): Maximum values for input data normalization.

    Returns:
        torch.Tensor: Best inverse solutions.
    """
    best_inv_x = []

    for i in range(len(y_all)):
        y_target = y_all[i, 0]
        x_true = x_all[i]

        # Perform localization
        optimized_x = perform_localization(y_target, x_true, model)

        # Find the best match
        wt_error = torch.sum((optimized_x - x_true) ** 2, dim=1)
        best_idx = torch.argmin(wt_error)
        best_inv_x.append(optimized_x[best_idx].unsqueeze(0))

    best_inv_x = torch.cat(best_inv_x, dim=0)
    return best_inv_x


def visualize_posterior(samples, x_true, title):
    """
    Visualize the posterior distribution of the input parameters.

    Args:
        samples (torch.Tensor): Posterior samples.
        x_true (torch.Tensor): True input parameters.
        title (str): Title for the plot.
    """
    samples = samples.cpu().detach().numpy()
    x_true = x_true.cpu().numpy()

    plt.figure(figsize=(12, 8))
    for i in range(samples.shape[1]):
        plt.subplot(1, 3, i + 1)
        plt.hist(samples[:, i], bins=30, density=True, alpha=0.7)
        plt.axvline(x_true[i], color='red', linestyle='--', label='True Value')
        plt.xlabel(f'x[{i}]')
        plt.ylabel('Density')
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, title + '.png'))
    plt.close()


def main():
    """
    Main function to run the localization process.
    """

    try:
        # Load trained model
        model = load_trained_model(eval_model_path)

        # Load test dataset
        test_data = torch.load(test_data_path, weights_only=True)
        x_all = test_data['x'].to(device)
        y_all = test_data['y'].to(device)
        min_x = test_data['min_x'].to(device)
        max_x = test_data['max_x'].to(device)

        # Evaluate localization
        best_inv_x = evaluate_localization(model, x_all, y_all, min_x, max_x)

        # Denormalize data
        x_all_denormalized = inverse_minmax_normal(x_all.cpu(), min_x.cpu(), max_x.cpu()).numpy()
        best_inv_x_denormalized = inverse_minmax_normal(best_inv_x.cpu(), min_x.cpu(), max_x.cpu()).numpy()

        # Calculate R² scores
        r2_inv = r2_score(x_all_denormalized, best_inv_x_denormalized)
        r2_inv_1 = r2_score(x_all_denormalized[:, 0], best_inv_x_denormalized[:, 0])
        r2_inv_2 = r2_score(x_all_denormalized[:, 1], best_inv_x_denormalized[:, 1])
        r2_inv_3 = r2_score(x_all_denormalized[:, 2], best_inv_x_denormalized[:, 2])
        print(f"\nR² scores:")
        print(f"Overall: {r2_inv:.3f}")
        print(f"material component: {r2_inv_1:.3f}")
        print(f"screw speed: {r2_inv_2:.3f}")
        print(f"temperature: {r2_inv_3:.3f}")

        # Visualize posterior distributions
        for i in range(min(3, len(y_all))):
            y_target = y_all[i, 0]
            x_true = x_all[i]
            samples = sample_posterior(y_target, 5000, model)
            visualize_posterior(samples, x_true, f'Posterior Distribution Sample {i + 1}')

        # Record processing time
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error during localization: {str(e)}")


if __name__ == "__main__":
    start_time = time.time()
    main()
