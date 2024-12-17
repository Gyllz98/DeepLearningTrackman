import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import stats
# Constants
DATA_ROOT = Path("/dtu-compute/02456-p4-e24/data")
BATCH_SIZE = 1  # Keep as 1 for accurate timing
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
NUM_RUNS = 100

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr
from SearchNet import SearchNet

def measure_inference_times(model, loader, num_runs=NUM_RUNS):
    model.eval()
    times = []
    
    # Warm-up runs
    print("Performing warm-up runs...")
    with torch.no_grad():
        for _ in range(10):
            data = next(iter(loader))
            spectrogram = data["spectrogram"].to(DEVICE)
            _ = model(spectrogram)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"Measuring inference time over {num_runs} runs...")
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_runs:
                break
                
            spectrogram = data["spectrogram"].to(DEVICE)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(spectrogram)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_runs} samples")
    
    return np.array(times)

def plot_inference_distribution(baseline_times, final_times, save_path=None):
    # Double all font sizes
    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 40,
        'axes.labelsize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28
    })
    
    plt.figure(figsize=(12, 8))
    
    # Set background color and grid
    plt.gca().set_facecolor('white')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate kernel density estimation
    baseline_kernel = stats.gaussian_kde(baseline_times)
    final_kernel = stats.gaussian_kde(final_times)
    
    # Create x points for evaluation
    x_eval = np.linspace(min(min(baseline_times), min(final_times)),
                        max(max(baseline_times), max(final_times)),
                        200)
    
    # Plot density curves with filled areas
    plt.fill_between(x_eval, baseline_kernel(x_eval), alpha=0.3, color='blue', label='Baseline Model')
    plt.plot(x_eval, baseline_kernel(x_eval), color='blue', linewidth=2)

    plt.fill_between(x_eval, final_kernel(x_eval), alpha=0.3, color='red', label='Final Model')
    plt.plot(x_eval, final_kernel(x_eval), color='red', linewidth=2)
    
    # Add mean lines
    plt.axvline(np.mean(baseline_times), color='blue', linestyle='--', linewidth=2)
    plt.axvline(np.mean(final_times), color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reset the rc parameters to default after plotting
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Print statistics
    print("\nInference Time Statistics:")
    print(f"{'Metric':<15} {'Baseline (ms)':<15} {'Final (ms)':<15}")
    print("-" * 45)
    print(f"{'Mean':<15} {np.mean(baseline_times):<15.4f} {np.mean(final_times):<15.4f}")
    print(f"{'Std Dev':<15} {np.std(baseline_times):<15.4f} {np.std(final_times):<15.4f}")
    print(f"{'Median':<15} {np.median(baseline_times):<15.4f} {np.median(final_times):<15.4f}")
    print(f"{'Speedup':<15} {np.mean(baseline_times)/np.mean(final_times):.2f}x")

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
    
    # Load final model
    final_model = SearchNet(
        alpha=0.1,  # Replace with your best model's parameters
        beta=1.16,
        gamma_conv=0.75,
        gamma_fc=0.75
    ).to(DEVICE)
    final_model.load_state_dict(torch.load("/zhome/b3/8/148387/trackman_project/models/final_model_final_model"))
    
    # Measure inference times
    baseline_times = measure_inference_times(baseline_model, val_loader)
    final_times = measure_inference_times(final_model, val_loader)
    
    # Plot and save results
    plot_inference_distribution(
        baseline_times,
        final_times,
        save_path="inference_time_distribution.png"
    )