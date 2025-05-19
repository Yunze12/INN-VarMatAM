import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from train_config import *
from data_processing import load_and_preprocess_data, create_dataset
from losses import loss_forward_fit, loss_forward_mmd, loss_backward_mmd, loss_reconstruction, noise_batch
from model import INN


def optim_step(optimizer):
    optimizer.step()
    optimizer.zero_grad()


def split_data(data_set, n_splits=7):
    """
    Split the dataset into a test set and a training set, and then further split
    the training set into training and validation sets using 5 - fold cross - validation.

    Args:
        data_set (torch.utils.data.Dataset): Full dataset.
        n_splits (int, optional): Number of folds. Defaults to 7.

    Returns:
        tuple: A tuple containing the test dataset, a list of training datasets,
               and a list of validation datasets for each fold.
    """
    # Split into training and test sets
    labels = data_set.x[:, 0].numpy()
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=10, random_state=42)
    for train_index, test_index in sss_test.split(data_set.x.numpy(), labels):
        test_dataset = Subset(data_set, test_index)
        train_subset = Subset(data_set, train_index)

    # Further split training set into training and validation sets
    train_datasets = []
    val_datasets = []
    sss_val = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    train_subset_data = data_set[train_subset.indices]
    labels_train = train_subset_data[0][:, 0].numpy()
    le = LabelEncoder()
    labels_train = le.fit_transform(labels_train)

    for train_idx, val_idx in sss_val.split(train_subset_data[0].numpy(), labels_train):
        train_datasets.append(Subset(train_subset, train_idx))
        val_datasets.append(Subset(train_subset, val_idx))

    return test_dataset, train_datasets, val_datasets


def train_model():
    """
    Train the INN model with cross-validation.

    This function loads data, splits it into training/validation sets,
    trains the model on each fold, and saves the best models.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        # Load and preprocess data
        normal_x, normal_y, min_x, max_x, min_y, max_y = load_and_preprocess_data(data_file_path)

        # Create dataset and dataloader
        dataset = create_dataset(normal_x, normal_y)
        test_dataset, train_datasets, val_datasets = split_data(dataset)

        # Save test dataset
        torch.save({
            'x': test_dataset.dataset.x[test_dataset.indices],
            'y': test_dataset.dataset.y[test_dataset.indices],
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }, os.path.join(test_dataset_path, 'test_dataset.pth'))

        # Track performance across folds
        train_losses_fold = []
        val_losses_fold = []

        # Train on each fold
        for fold, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
            logging.info(f"\nFold {fold + 1}/{len(train_datasets)} training starts")

            # Create data loaders
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

            # Initialize model, optimizer, and scheduler
            inn = INN().to(device)
            optimizer = torch.optim.Adam(
                inn.parameters(),
                lr=lr_init,
                betas=adam_betas,
                eps=1e-6,
                weight_decay=l2_weight_reg
            )
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

            # Track losses for early stopping
            train_losses = []
            val_losses = []

            for epoch in range(num_epochs):
                # Training phase
                inn.train()
                epoch_loss = 0.0
                for inputs, targets in train_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Forward pass
                    outputs = inn.forward(inputs)

                    # Compute losses
                    losses = []
                    if train_forward_fit:
                        losses.append(loss_forward_fit(outputs, targets))
                    if train_forward_mmd:
                        losses.append(loss_forward_mmd(outputs, targets))
                    if train_backward_mmd:
                        losses.append(loss_backward_mmd(inputs, targets))
                    if train_reconstruction:
                        losses.append(loss_reconstruction(outputs.data, inputs))

                    # Backpropagation
                    total_loss = sum(losses)
                    total_loss.backward()
                    optim_step(optimizer)

                    train_losses.append(total_loss.item())

                # Log training loss
                logging.info(f"Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.6f}")

                # Learning rate scheduling
                if len(train_losses) >= 10:
                    now_loss = train_losses[-5:]
                    now_average_loss = sum(now_loss) / len(now_loss)
                    scheduler.step(int(now_average_loss))

                # Validation phase
                inn.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_dataloader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        outputs = inn.forward(inputs)

                        # Compute validation losses
                        losses = []
                        if train_forward_fit:
                            losses.append(loss_forward_fit(outputs, targets))
                        if train_forward_mmd:
                            losses.append(loss_forward_mmd(outputs, targets))
                        if train_backward_mmd:
                            losses.append(loss_backward_mmd(inputs, targets))
                        if train_reconstruction:
                            losses.append(loss_reconstruction(outputs.data, inputs))

                        val_loss = sum(losses)
                        val_losses.append(val_loss.item())

                # Log validation loss
                logging.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.6f}")

                # Save model checkpoint
                if (epoch + 1) % 100 == 0:
                    save_path = os.path.join(model_path, f'modelfold{fold + 1}epoch{epoch + 1}.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': inn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'min_x': min_x,
                        'max_x': max_x,
                        'min_y': min_y,
                        'max_y': max_y
                    }, save_path)
                    logging.info(f"Model saved to {save_path}")

            # Save final model for this fold
            save_path = os.path.join(model_path, f'modelfold{fold + 1}epoch{num_epochs}.pth')
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': inn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            }, save_path)
            logging.info(f"Final model saved to {save_path}")

            # Track fold performance
            train_losses_fold.append(train_losses[-1])
            val_losses_fold.append(val_loss)
            logging.info(f"Fold {fold + 1} complete. Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")

        # Log overall performance
        logging.info("\nTraining complete. Cross-validation results:")
        logging.info(f"Average Training Loss: {np.mean(train_losses_fold):.6f} ± {np.std(train_losses_fold):.6f}")
        logging.info(f"Average Validation Loss: {np.mean(val_losses_fold):.6f} ± {np.std(val_losses_fold):.6f}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    train_model()









