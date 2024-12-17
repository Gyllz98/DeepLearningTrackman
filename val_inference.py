import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Reuse your constants and imports from the training script
DATA_ROOT = Path("/dtu-compute/02456-p4-e24/data")
BATCH_SIZE = 500
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

# Import your custom modules (adjust paths as needed)
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr
from SearchNet import SearchNet

def load_model_and_compute_residuals(model, loader):
    model.eval()
    residuals = []
    
    with torch.no_grad():
        for data in loader:
            spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)
            outputs = model(spectrogram)
            batch_residuals = (outputs.squeeze() - target).cpu().numpy()
            residuals.extend(batch_residuals)
    
    return np.array(residuals)

def plot_residuals_distribution(baseline_residuals, final_residuals, save_path=None):
    # Double all font sizes
    plt.rcParams.update({
        'font.size': 28,          # Default font size
        'axes.titlesize': 40,     # Title font size
        'axes.labelsize': 32,     # Axis label font size
        'xtick.labelsize': 28,    # X-axis tick labels
        'ytick.labelsize': 28,    # Y-axis tick labels
        'legend.fontsize': 28     # Legend font size
    })
    
    plt.figure(figsize=(12, 8))
    
    # Set background color and grid
    plt.gca().set_facecolor('white')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate kernel density estimation
    baseline_kernel = stats.gaussian_kde(baseline_residuals)
    final_kernel = stats.gaussian_kde(final_residuals)
    
    # Create x points for evaluation
    x_eval = np.linspace(min(min(baseline_residuals), min(final_residuals)),
                        max(max(baseline_residuals), max(final_residuals)),
                        200)
    
    # Plot density curves with filled areas
    plt.fill_between(x_eval, baseline_kernel(x_eval), alpha=0.3, color='blue', label='Baseline Model')
    plt.plot(x_eval, baseline_kernel(x_eval), color='blue', linewidth=2)
    
    plt.fill_between(x_eval, final_kernel(x_eval), alpha=0.3, color='red', label='Final Model')
    plt.plot(x_eval, final_kernel(x_eval), color='red', linewidth=2)
    
    # Add mean lines
    plt.axvline(np.mean(baseline_residuals), color='blue', linestyle='--', linewidth=2)
    plt.axvline(np.mean(final_residuals), color='red', linestyle='--', linewidth=2)

    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reset the rc parameters to default after plotting
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    # Set up validation dataset
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name
    
    transform_val = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "validation"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])
    
    dataset_val = SpectrVelCNNRegr.dataset(
        data_dir=data_dir / "validation",
        stmf_data_path=DATA_ROOT / "stmf_data_3.csv",
        transform=transform_val
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1
    )
    
    # Load baseline model
    baseline_model = SpectrVelCNNRegr().to(DEVICE)
    baseline_model.load_state_dict(torch.load("/zhome/b3/8/148387/trackman_project/models/baseline_model_baseline_model"))
    
    # Load final model (adjust path and parameters as needed)
    # You'll need to provide the correct parameters from your best model
    final_model = SearchNet(
        alpha=0.1,  # Replace with your best model's parameters
        beta=1.16,   # Replace with your best model's parameters
        gamma_conv=0.75,  # Replace with your best model's parameters
        gamma_fc=0.75     # Replace with your best model's parameters
    ).to(DEVICE)
    final_model.load_state_dict(torch.load("/zhome/b3/8/148387/trackman_project/models/final_model_final_model"))
    
    # Compute residuals
    baseline_residuals = load_model_and_compute_residuals(baseline_model, val_loader)
    final_residuals = load_model_and_compute_residuals(final_model, val_loader)
    
    # Plot and save results
    plot_residuals_distribution(
        baseline_residuals,
        final_residuals,
        save_path="residuals_distribution.png"
    )