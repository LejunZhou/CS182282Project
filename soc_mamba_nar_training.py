import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
from time import perf_counter

import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from mambapy.mamba import Mamba, MambaConfig
from tqdm import tqdm  # <-- added tqdm

# Automated device selection based on available backends
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and False
    else "cpu"
)

print(f"> Using {device} device")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(filepath, model, scheduler, optimizer):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)
        loaded_epoch = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['model_state'])
        if scheduler is not None:
            scheduler.load_state_dict(loaded_checkpoint['scheduler_state'])
        if optimizer is not None:
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        print("> Loaded model")
        return True, loaded_epoch, model, scheduler, optimizer
    except Exception as e:
        print(f"> Cannot load model: {e}")
        return False, 0, model, scheduler, optimizer

class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MambaSOCModel(torch.nn.Module):
    def __init__(self, d_model=64, n_layers=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(2, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.output_head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)              # (B, L, D)
        x = self.mamba(x)                   # (B, L, D)
        return self.output_head(x).squeeze(-1)  # (B, L)

def train():
    # Training parameters
    epochs = 11
    batch_size = 32
    learning_rate = 1e-3
    model_path = f'saves_nar/model.pth'
    backup_path = f"saves_nar/model-b.pth"

    # Load preprocessed dataset
    with open("data/90_ SOH/battery_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32)

    dataset = BatteryDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MambaSOCModel(d_model=64, n_layers=4).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.1, patience=2
    )

    # Load checkpoint if available
    _, epoch, model, scheduler, optim = load_checkpoint(model_path, model, scheduler, optim)

    # Training loop
    t0_start = perf_counter()
    for z in range(epoch, epochs):
        avg_loss = 0
        print(f"\n> Epoch {z+1}/{epochs}")
        t2_start = perf_counter()

        progress_bar = tqdm(train_loader, desc=f"Epoch {z+1}/{epochs}", leave=False)
        for input, target in progress_bar:
            model.train()
            input = input.to(device)
            target = target.to(device)

            pred = model(input)  # (B, L)
            loss = F.mse_loss(pred, target)
            avg_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            progress_bar.set_postfix(loss=loss.item())

        t2_stop = perf_counter()
        print(f"> Epoch time: {t2_stop - t2_start:.3f} seconds | Avg Loss: {avg_loss/len(train_loader):.5f}")

        scheduler.step(avg_loss/len(train_loader))

        checkpoint = {
            'epoch': z + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optim.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        if backup_path is not None and os.path.isfile(model_path):
            shutil.copyfile(model_path, backup_path)
        torch.save(checkpoint, model_path)

    t0_stop = perf_counter()
    print(f"\n> Finished training in: {t0_stop - t0_start:.3f} seconds")

def prepare_folders():
    try:
        os.makedirs("./saves_nar/")
    except FileExistsError:
        pass

if __name__ == "__main__":
    seed_everything(534)
    prepare_folders()
    train()