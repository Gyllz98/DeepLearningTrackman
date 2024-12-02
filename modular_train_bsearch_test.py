from pathlib import Path

from numpy import log10
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import wandb

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr, YourModel, weights_init_uniform_rule
from binary_search import *
from check_model_complexity import *

# GROUP NUMBER
GROUP_NUMBER = 42

# CONSTANTS TO MODIFY AS YOU WISH
MODEL = YourModel
LEARNING_RATE = 10**-5
EPOCHS = 100 # the model converges in test perfermance after ~250-300 epochs
BATCH_SIZE = 10
NUM_WORKERS = 4
OPTIMIZER = torch.optim.SGD
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# DEVICE = "cpu"

# You can set the model path name in case you want to keep training it.
# During the training/testing loop, the model state is saved
# (only the best model so far is saved)
LOAD_MODEL_FNAME = None
LOAD_MODEL_PATH = f"model_{MODEL.__name__}_noble-meadow-16"

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)


def train_one_epoch(loss_fn, model, train_data_loader):
    running_loss = 0.
    last_loss = 0.
    total_loss = 0.

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(spectrogram)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(), target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        total_loss += loss.item()
        if i % train_data_loader.batch_size == train_data_loader.batch_size - 1:
            last_loss = running_loss / train_data_loader.batch_size # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return total_loss / (i+1)

def evaluate_loss(loss_fn, model, data_loader):
    """Evaluate the model's performance on validation/test data."""
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    return avg_loss

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TRAIN_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    TEST_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "test"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    dataset_train = MODEL.dataset(data_dir= data_dir / "train",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TRAIN_TRANSFORM)

    dataset_test = MODEL.dataset(data_dir= data_dir / "test",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TEST_TRANSFORM)
    
    train_data_loader = DataLoader(dataset_train, 
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS)
    test_data_loader = DataLoader(dataset_test,
                                  batch_size=500,
                                  shuffle=False,
                                  num_workers=1)
    
    # If you want to keep training a previous model
    if LOAD_MODEL_FNAME is not None:
        model = MODEL().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / LOAD_MODEL_FNAME))
        model.eval()
    else:
        model = MODEL().to(DEVICE)
        model.apply(weights_init_uniform_rule)

    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    
    # Set up wandb for reporting
    wandb.init(
        project=f"trackman-project",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": MODEL.__name__,
            "dataset": MODEL.dataset.__name__,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "transform": "|".join([str(tr).split(".")[1].split(" ")[0] for tr in dataset_train.transform.transforms]),
            "optimizer": OPTIMIZER.__name__,
            "loss_fn": model.loss_fn.__name__,
            "nfft": NFFT
        }
    )

    # Define model output to save weights during training
    MODEL_DIR.mkdir(exist_ok=True)
    model_name = f"model_{MODEL.__name__}_{wandb.run.name}"
    model_path = MODEL_DIR / model_name

    # import pdb; pdb.set_trace()

    # Calculate initial max parameters
    total_params = 1e6 # CHANGE THIS (check_model_complexity)

    # Run one training cycle to calculate the initial loss
    initial_loss = evaluate_loss(model, train_data_loader)

    # Initialize binary search
    binary_search = BinarySearch(
        init_max_params=total_params,
        init_right_loss=initial_loss,
        model=model,
        tolerance=0.05  # Adjust if necessary
    )

    ## TRAINING LOOP
    epoch_number = 0 # CHANGE THIS 
    best_vloss = 1_000_000.
    
    while True:
        print("Starting training with current model parameters...")

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Training mode
            model.train(True)
            avg_loss = 0.0
            for i, data in enumerate(train_data_loader, 0):
                inputs, targets = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(inputs)
                loss = model.loss_fn(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()

            avg_loss /= len(train_data_loader)

            # Calculate metrics
            rmse = avg_loss**(1 / 2)
            log_rmse = log10(rmse)

            # Validation phase
            running_test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(test_data_loader):
                    spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                    test_outputs = model(spectrogram)
                    test_loss = model.loss_fn(test_outputs.squeeze(), target)
                    running_test_loss += test_loss.item()

            avg_test_loss = running_test_loss / (i + 1)
            test_rmse = avg_test_loss**(1 / 2)
            log_test_rmse = log10(test_rmse)

            print('LOSS train {} ; LOSS test {}'.format(avg_loss, avg_test_loss))

            # Log metrics to WandB
            wandb.log({
                "loss": avg_loss,
                "rmse": rmse,
                "log_rmse": log_rmse,
                "test_loss": avg_test_loss,
                "test_rmse": test_rmse,
                "log_test_rmse": log_test_rmse,
            })

            # Track best performance
            if avg_test_loss < best_vloss:
                best_vloss = avg_test_loss
                torch.save(model.state_dict(), "best_model.pth")

            epoch_number += 1

        # Perform a binary search step
        print("Performing binary search...")
        bsearch_finish, next_params = binary_search.search_next_params(avg_test_loss)

        # Check the tolerance condition for breaking the loop
        if bsearch_finish == 0:
            print("Binary search completed.")
            break

        # Update model architecture dynamically if necessary
        print(f"Updating model to {next_params} parameters...")
        model = SpectrVelCNNRegr()  # Create a new model instance
        model.apply(weights_init_uniform_rule)
        model = model.to(DEVICE)

        # Reinitialize optimizer with the new model
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training and binary search completed.")
    wandb.finish()
