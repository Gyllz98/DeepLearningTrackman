# Getting started

After logging in to the DTU cluster and accessing your project folder, switch to an interactive node:

```bash
voltash
```

This will keep IT support off your backs, so you don't run stuff on the login nodes.

## Data access

Data is available on the cluster in the directory

```
/dtu-compute/maalhe/tm/02456/data/
```

where you will find a metadata file `stmf_data_3.csv` with targets and spectrogram data folder `/data_fft-512_tscropwidth-150-200_vrcropwidth-60-15`, which itself has subdirectories with `train` and `test` data (and `validation`, but disregard that for now).

These folders are automatically accessed from the training script, so you don't have to do anythig in this regard.

## Dependencies

In your project folder on the cluster, create a virtual environment and activate it:

```bash
python3 -m venv 02456_grp_99_venv
source 02456_grp_99_venv/bin/activate
```

Then, install the required packages using `pip` and the `requirements.txt` file:

```bash
pip install -r scripts/requirements.txt
```

## Training the baseline model

Run the training/testing script `scripts/modular_train_test.py` to trian the baseline model for 1 epoch:

```bash
python scripts/modular_train_test.py
```

to verify that the baseline model and training script works as intended.

# Training models to convergence

Using the interactive client to run the training script it fine, if you are training one (or a couple of) epoch(s) to ensure that things are working.

However, when you transition to training models to convergence, you are expected to submit your jobs, putting you in a queue with the others on the cluster. I have prepared a jobscript for you: `scripts/modular_train_test_job.sh`, which you run like this:

```bash
bsub < scripts/modular_train_test_job.sh
```

*Please note* that the job script queues your job and references your python training script (`scripts/modular_train_test.py`). However, that python script is not interpreted until your job starts! This means that any changes you make to the script after submitting your job will also be interpreted!

## Monitoring your model(s)

I suggest you use *wieghts & biases* to monitor your models. See https://wandb.ai/site/ for details. You can create an account with your student credentials and then configure it for your user on the cluster.

There is a block in the training loop which sets up the reporting configuration

```python
# Set up wandb for reporting
wandb.init(
    project=f"02456_group_{GROUP_NUMBER}",
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
```

and then you can monitor the loss (and any other parameters of interest) interactively online.

You can also access the data manually in the automatically created directory `/wandb` if you don't want to use the online dashboard.