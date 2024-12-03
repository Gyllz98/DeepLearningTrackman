from pathlib import Path
from numpy import log10
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from loss import mse_loss
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr
from SearchNet import SearchNet, find_scaling_factors_grid_search, calculate_params_precise
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add early stopping parameters
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 0.0001

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# GROUP NUMBER
GROUP_NUMBER = 42

# CONSTANTS TO MODIFY AS YOU WISH
LEARNING_RATE = 10**-5
EPOCHS = 200
BATCH_SIZE = 16
NUM_WORKERS = 4
OPTIMIZER = torch.optim.SGD
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

def train_one_epoch(loss_fn, model, train_data_loader, optimizer):
    model.train()
    total_loss = 0.

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(spectrogram)
        loss = loss_fn(outputs.squeeze(), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 10 == 9:  # Print every 10 batches
            print(f'  batch {i + 1} loss: {loss.item():.4f}')

    return total_loss / (i + 1)

def evaluate_model(model, eval_loader):
    model.eval()
    total_loss = 0.
    num_batches = 0  # Track number of batches explicitly
    
    with torch.no_grad():
        for data in eval_loader:
            spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)
            outputs = model(spectrogram)
            loss = mse_loss(outputs.squeeze(), target)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(1, num_batches)  # Avoid division by zero if loader is empty


if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    # Define transforms
    transform_train = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])

    transform_val = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "validation"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])

    transform_test = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "test"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])

    # Load datasets
    dataset_train = SpectrVelCNNRegr.dataset(data_dir=data_dir / "train",
                                          stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                          transform=transform_train)
    
    dataset_val = SpectrVelCNNRegr.dataset(data_dir=data_dir / "validation",
                                        stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                        transform=transform_val)
    
    dataset_test = SpectrVelCNNRegr.dataset(data_dir=data_dir / "test",
                                         stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                         transform=transform_test)

    # Create combined dataset for full training
    dataset_train_test = ConcatDataset([dataset_train, dataset_test])

    # Create dataloaders
    train_test_loader = DataLoader(dataset_train_test, 
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS)
    
    train_loader = DataLoader(dataset_train, 
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS)
    
    val_loader = DataLoader(dataset_val,
                          batch_size=500,  # Full validation set
                          shuffle=False,
                          num_workers=1)
    
    test_loader = DataLoader(dataset_test,
                           batch_size=500,  # Full test set
                           shuffle=False,
                           num_workers=1)

    MODEL_DIR.mkdir(exist_ok=True)

    # Phase 1: Train baseline model on train + test, evaluate on val
    print("\nPhase 1: Training Baseline Model")
    wandb.init(
        project="trackman-project",
        name="baseline_model",
        config={
            "phase": "baseline",
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "architecture": SpectrVelCNNRegr.__name__,
            "optimizer": OPTIMIZER.__name__,
        }
    )

    baseline_model = SpectrVelCNNRegr().to(DEVICE)
    baseline_optimizer = OPTIMIZER(baseline_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    baseline_scheduler = ReduceLROnPlateau(baseline_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f'Baseline Training - EPOCH {epoch + 1}:')
        train_loss = train_one_epoch(mse_loss, baseline_model, train_test_loader, baseline_optimizer)
        val_loss = evaluate_model(baseline_model, val_loader)
        baseline_scheduler.step(val_loss)
        
        train_rmse = train_loss ** 0.5
        val_rmse = val_loss ** 0.5
        
        wandb.log({
            "baseline/train_rmse": train_rmse,
            "baseline/val_rmse": val_rmse,
            "baseline/train_log_rmse": log10(train_rmse),
            "baseline/val_log_rmse": log10(val_rmse),
            "baseline/epoch": epoch,
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(baseline_model.state_dict(), MODEL_DIR / f"baseline_model_{wandb.run.name}")
        
    # Phase 2: Binary search using SearchNet and RMSE-based decisions
    print("\nPhase 2: Binary Search for Optimal Parameters")
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    left = int(baseline_params * 0.01)
    right = baseline_params
    tolerance = baseline_params * 0.001
    best_test_rmse = float('inf')
    best_config = None
    best_model_state = None

    # For tracking model performance
    model_performances = []

    wandb.finish()  # Finish baseline run

    # Start binary search wandb run
    wandb.init(
        project="trackman-project",
        name="binary_search",
        config={
            "phase": "search",
            "max_parameters": baseline_params,
            "min_parameters": left,
            "tolerance": tolerance,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_delta": EARLY_STOPPING_DELTA,
        }
    )

    # Create a new wandb Table to log model performances
    performance_table = wandb.Table(columns=["parameters", "rmse", "architecture"])

    # Add baseline to performance tracking after wandb init
    baseline_rmse = (best_val_loss ** 0.5)  # Convert MSE to RMSE
    model_performances.append({
        'parameters': baseline_params,
        'rmse': baseline_rmse,
        'architecture': 'Baseline SpectrVelCNNRegr'
    })
    performance_table.add_data(baseline_params, baseline_rmse, 'Baseline SpectrVelCNNRegr')

    search_iteration = 0
    target_rmse = None  # Will be set based on first iteration

    while (right - left) > tolerance:
        search_iteration += 1
        mid = (left + right) // 2
        print(f"\nBinary Search Iteration {search_iteration}")
        print(f"Searching for configuration with ~{mid:,d} parameters")
        print(f"Current bounds: [{left:,d}, {right:,d}]")
        
        # Find scaling factors for target parameter count
        result = find_scaling_factors_grid_search(
            target_params=mid,
            tolerance=0.05,
            alpha_range=(0.1, 3, 30),
            beta_range=(0.75, 1.5, 30)
        )
        
        if result is None:
            print(f"Could not find valid configuration for {mid} parameters")
            right = mid
            continue
            
        alpha, beta, gamma_conv, gamma_fc, expected_params, _, _ = result
        
        # Train model with current configuration
        search_model = SearchNet(
            alpha=alpha, 
            beta=beta, 
            gamma_conv=gamma_conv, 
            gamma_fc=gamma_fc
        ).to(DEVICE)
        
        search_optimizer = OPTIMIZER(search_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        search_scheduler = ReduceLROnPlateau(search_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        best_iteration_rmse = float('inf')
        early_stopping = EarlyStopping()
        
        for epoch in range(EPOCHS):
            print(f'Search Training (params={expected_params}) - EPOCH {epoch + 1}:')
            train_loss = train_one_epoch(mse_loss, search_model, train_loader, search_optimizer)
            test_loss = evaluate_model(search_model, test_loader)
            search_scheduler.step(test_loss)
            
            train_rmse = train_loss ** 0.5
            test_rmse = test_loss ** 0.5
            
            wandb.log({
                f"search_iter_{search_iteration}/train_rmse": train_rmse,
                f"search_iter_{search_iteration}/test_rmse": test_rmse,
                f"search_iter_{search_iteration}/train_log_rmse": log10(train_rmse),
                f"search_iter_{search_iteration}/test_log_rmse": log10(test_rmse),
                f"search_iter_{search_iteration}/epoch": epoch,
                "binary_search/current_params": expected_params,
                "binary_search/left_bound": left,
                "binary_search/right_bound": right
            })
            
            # Early stopping check
            early_stopping(test_rmse)
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            if test_rmse < best_iteration_rmse:
                best_iteration_rmse = test_rmse
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_config = (alpha, beta, gamma_conv, gamma_fc)
                    best_model_state = search_model.state_dict().copy()

        # After training each configuration
        model_performances.append({
            'parameters': expected_params,
            'rmse': best_iteration_rmse,
            'architecture': f"SearchNet(α={alpha:.2f}, β={beta:.2f}, γ_conv={gamma_conv:.2f}, γ_fc={gamma_fc:.2f})"
        })

        # Log to wandb table
        performance_table.add_data(expected_params, best_iteration_rmse, 
                                f"SearchNet(α={alpha:.2f}, β={beta:.2f}, γ_conv={gamma_conv:.2f}, γ_fc={gamma_fc:.2f})")
        
        # Set target RMSE based on first iteration
        if search_iteration == 1:
            target_rmse = best_iteration_rmse * 1.05  # Allow 5% degradation
        
        # Update binary search bounds based on RMSE performance
        if best_iteration_rmse > target_rmse:
            left = mid  # Need more parameters to improve performance
        else:
            right = mid  # Can try fewer parameters
            
        wandb.log({
            "binary_search/iteration": search_iteration,
            "binary_search/best_test_rmse": best_test_rmse,
            "binary_search/target_rmse": target_rmse
        })

    # Create scatter plot using wandb
    wandb.log({
        "model_comparison": wandb.plot.scatter(
            performance_table, 
            "parameters", 
            "rmse",
            title="Model Size vs Performance"
        )
    })

    # Also save the data for later analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    performance_data = {
        'experiment_config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_delta': EARLY_STOPPING_DELTA,
            'total_parameters_baseline': baseline_params,
            'total_iterations': search_iteration
        },
        'models': model_performances,
        'best_model': {
            'parameters': best_config[0] if best_config else None,
            'rmse': best_test_rmse,
            'config': {
                'alpha': best_config[0],
                'beta': best_config[1],
                'gamma_conv': best_config[2],
                'gamma_fc': best_config[3]
            } if best_config else None
        }
    }

    with open(MODEL_DIR / f'model_performances_{timestamp}.json', 'w') as f:
        json.dump(performance_data, f, indent=2)

    # Print summary of all configurations
    print("\nModel Performance Summary:")
    print(f"{'Parameters':>12} | {'RMSE':>10} | Architecture")
    print("-" * 70)
    for perf in sorted(model_performances, key=lambda x: x['parameters']):
        print(f"{perf['parameters']:12,d} | {perf['rmse']:10.4f} | {perf['architecture']}")
    
    wandb.finish()

    # Phase 3: Final training with best configuration
    print("\nPhase 3: Final Training with Best Configuration")
    if best_config is not None:
        alpha, beta, gamma_conv, gamma_fc = best_config
        wandb.init(
            project="trackman-project",
            name="final_model",
            config={
                "phase": "final",
                "alpha": alpha,
                "beta": beta,
                "gamma_conv": gamma_conv,
                "gamma_fc": gamma_fc,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "early_stopping_delta": EARLY_STOPPING_DELTA,
            }
        )

        final_model = SearchNet(
            alpha=alpha,
            beta=beta,
            gamma_conv=gamma_conv,
            gamma_fc=gamma_fc
        ).to(DEVICE)
        
        if best_model_state is not None:
            final_model.load_state_dict(best_model_state)
        
        final_optimizer = OPTIMIZER(final_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        best_final_val_loss = float('inf')
        early_stopping = EarlyStopping()

        for epoch in range(EPOCHS):
            print(f'Final Training - EPOCH {epoch + 1}:')
            train_loss = train_one_epoch(mse_loss, final_model, train_test_loader, final_optimizer)
            val_loss = evaluate_model(final_model, val_loader)
            final_scheduler.step(val_loss)
            
            train_rmse = train_loss ** 0.5
            val_rmse = val_loss ** 0.5
            
            wandb.log({
                "final/train_rmse": train_rmse,
                "final/val_rmse": val_rmse,
                "final/train_log_rmse": log10(train_rmse),
                "final/val_log_rmse": log10(val_rmse),
                "final/epoch": epoch,
            })
            
            # Early stopping check
            early_stopping(val_rmse)
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            if val_loss < best_final_val_loss:
                best_final_val_loss = val_loss
                torch.save(final_model.state_dict(), MODEL_DIR / f"final_model_{wandb.run.name}")

        wandb.finish()