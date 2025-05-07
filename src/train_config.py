import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='w')
    ]
)

# Set the device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Hyperparameters and configuration settings
init_scale = 0.10
mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = False
batch_size = 60
lambd_fit_forw = 5.
lambd_mmd_forw = 1.
lambd_reconstruct = 1e-7
lambd_mmd_back = 1
lambd_max_likelihood = 1.
y_uncertainty_sigma = 0.12 * 4
train_forward_fit = True
train_forward_mmd = True
train_backward_mmd = True
train_reconstruction = False
train_max_likelihood = False
add_y_noise = 5e-2
add_z_noise = 2e-2
add_pad_noise = 1e-2
ndim_y = 1
ndim_z = 2
data_file_path = 'C:/Users/Lcxg-6/Desktop/inn2/data/data.xlsx'
test_dataset_path = 'C:/Users/Lcxg-6/Desktop/inn2/data/'
test_data_path = 'C:/Users/Lcxg-6/Desktop/inn2/data/test_dataset.pth'
model_path = 'C:/Users/Lcxg-6/Desktop/inn2/results/trained_model/'
eval_model_path = 'C:/Users/Lcxg-6/Desktop/inn2/results/trained_model/modelfold1epoch8000.pth'
figure_path = 'C:/Users/Lcxg-6/Desktop/inn2/results/figure/'

# Training parameters
seed = 61
num_epochs = 8000
final_decay = 0.02
lr_init = 5.0e-3
adam_betas = (0.9, 0.95)
l2_weight_reg = 1e-5
gamma = final_decay ** (1. / num_epochs)
