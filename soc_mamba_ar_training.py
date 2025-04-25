import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import random
from time import perf_counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from mambapy.mamba import Mamba, MambaConfig, RMSNorm

# ------------------ Model ------------------
from models import *

# ------------------ Dataset ------------------
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # (N, T, 2)
        self.y = y  # (N, T)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------ Utilities ------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_folders():
    try:
        os.makedirs("./saves_ar/")
    except FileExistsError:
        pass

# ------------------ Training ------------------
def train():
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() and False else
        "cpu"
    )
    print(f"> Using {device} device")

    # Parameters
    extra_epochs = 5
    batch_size = 32
    learning_rate = 5*1e-7
    d_model = 32
    n_layers = 2
    soc_window = 4
    lambda_smooth = 0.05  # Weight of the smoothness loss, you can tune this (0 if you don't want it)
    model_path = 'saves_ar/soc_mamba_v1/model.pth'
    backup_path = 'saves_ar/soc_mamba_v1/model-b.pth'
    teacher_forcing = 'partial'  # 'none' or 'partial'

    # Load data
    with open("data/90_ SOH/battery_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    X_raw = torch.tensor(data['X'], dtype=torch.float32)  # (N, T, 2)
    y_true = torch.tensor(data['y'], dtype=torch.float32) # (N, T)

    # Keep only 10% of data (quick training)
    N = X_raw.shape[0]
    num_keep = int(0.1 * N)
    indices = torch.randperm(N)[:num_keep]
    X_subset = X_raw[indices]
    y_subset = y_true[indices]

    padded_soc = F.pad(y_subset, (soc_window, 0))
    X_input = []
    for t in range(y_subset.size(1)):
        soc_window_slice = padded_soc[:, t:t+soc_window]
        i_u_t = X_subset[:, t, :]
        x_t = torch.cat([i_u_t, soc_window_slice], dim=1)
        X_input.append(x_t.unsqueeze(1))
    X_input = torch.cat(X_input, dim=1)  # (N, T, 2 + soc_window)

    dataset = BatteryDataset(X_input, y_subset)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model and optimizer
    model = SOC_Mamba_V1(d_model=d_model, n_layers=n_layers, soc_window=soc_window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    start_epoch = 0
    if os.path.exists(model_path):
        print(f"> Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        t0_start = perf_counter()

    for epoch in range(start_epoch, start_epoch + extra_epochs):  # continue 10 more epochs
        model.train()
        total_loss = 0
        print(f"\n> Epoch {epoch+1}/{start_epoch+extra_epochs}")
        t1_start = perf_counter()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+extra_epochs}", leave=False)
        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            
            if teacher_forcing == 'partial':
                B, T = yb.shape
                soc_window_size = model.soc_window
                pred_seq = []

                # Initialize soc_buffer with first several ground-truth SOC values
                soc_buffer = yb[:, :soc_window_size].detach().clone()  # shape (B, 5)

                # Start from t=5
                for t in range(soc_window_size, T):
                    xt = xb[:, t, :2]  # (B, 2)
                    xt_input = torch.cat([xt, soc_buffer], dim=1).unsqueeze(1)  # (B, 1, 2+5)
                    soc_t = model.step(xt_input)
                    pred_seq.append(soc_t)
                    soc_buffer = torch.cat([soc_buffer[:, 1:], soc_t.unsqueeze(1)], dim=1)

                # Stack predictions and compare against yb[:, 5:]
                pred = torch.stack(pred_seq, dim=1)  # shape (B, T - 5)
                loss = F.mse_loss(pred, yb[:, soc_window_size:])

                # Penalize large differences between consecutive SOC predictions
                smoothness_penalty = F.mse_loss(pred[:, 1:], pred[:, :-1])
                loss = loss + lambda_smooth * smoothness_penalty


            else:
                B, T = yb.shape
                soc_window_size = model.soc_window
                pred_seq = []
                soc_buffer = torch.zeros(B, soc_window_size).to(device)

                for t in range(T):
                    xt = xb[:, t, :2]  # (B, 2)
                    xt_input = torch.cat([xt, soc_buffer], dim=1).unsqueeze(1)  # (B, 1, 2+W)
                    soc_t = model.step(xt_input)
                    pred_seq.append(soc_t)
                    soc_buffer = torch.cat([soc_buffer[:, 1:], soc_t.unsqueeze(1)], dim=1)

                pred = torch.stack(pred_seq, dim=1)  # (B, T)

                loss = F.mse_loss(pred, yb)

                # Penalize large differences between consecutive SOC predictions
                smoothness_penalty = F.mse_loss(pred[:, 1:], pred[:, :-1], reduction='sum') / (pred.size(1) - 1)
                loss = loss + lambda_smooth * smoothness_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        t1_stop = perf_counter()
        avg_train_loss = total_loss / len(train_loader)
        print(f"> Epoch time: {t1_stop - t1_start:.2f}s | Avg Train Loss: {avg_train_loss:.5f}")

        # Validation (autoregressive, no teacher forcing)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)  # xb: (1, T, 3), yb: (1, T)
                B, T = yb.shape
                soc_window_size = model.soc_window
                pred_seq = []
                soc_buffer = torch.zeros(B, soc_window_size).to(device)

                for t in range(soc_window_size, T):
                    xt = xb[:, t, :2]  # I_t, U_t
                    xt_input = torch.cat([xt, soc_buffer], dim=1).unsqueeze(1)
                    soc_t = model.step(xt_input)  # (B,)
                    pred_seq.append(soc_t)
                    soc_buffer = torch.cat([soc_buffer[:, 1:], soc_t.unsqueeze(1)], dim=1)

                pred = torch.stack(pred_seq, dim=1)  # (B, T)
                val_loss += F.mse_loss(pred, yb[:, soc_window_size:], reduction='sum').item()

        avg_val_loss = val_loss / (len(val_set) * y_true.size(1))
        print(f"> Validation RMSE: {avg_val_loss ** 0.5:.5f}")

        scheduler.step(avg_train_loss)

        # Save model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        if backup_path is not None and os.path.isfile(model_path):
            shutil.copyfile(model_path, backup_path)
        torch.save(checkpoint, model_path)

    t0_stop = perf_counter()
    print(f"> Finished training in {t0_stop - t0_start:.2f} seconds")

# ------------------ Entry ------------------
if __name__ == "__main__":
    seed_everything(534)
    prepare_folders()
    train()