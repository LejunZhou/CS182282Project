import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import random
from time import perf_counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from mambapy.mamba import Mamba, MambaConfig, RMSNorm
from sklearn.model_selection import train_test_split

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
    extra_epochs = 20
    batch_size = 32
    learning_rate = 5*1e-5
    d_model = 32
    n_layers = 2
    soc_window = 5
    lambda_smooth = 0  # Weight of the smoothness loss, you can tune this (0 if you don't want it)
    noise_std = 0.01  # adjust as needed
    model_path = 'saves_ar/soc_mamba_v3/model.pth'
    backup_path = 'saves_ar/soc_mamba_v3/model-b.pth'
    teacher_forcing = 'true'  # 'true', 'partial', or 'false'
    validation = 'true'  # 'true' or 'false'

    # Load data
    with open("data/battery_dataset_combined.pkl", "rb") as f:
        data = pickle.load(f)

    # X_raw = torch.tensor(data['X'], dtype=torch.float32)  # (N, T, 2)
    # y_true = torch.tensor(data['y'], dtype=torch.float32) # (N, T)
    X_raw, y_true = data

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_true, test_size=0.3, random_state=666)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=666)

    # Create TensorDatasets
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_test, y_test)


    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # # Only use the first instance
    # val_subset = torch.utils.data.Subset(val_set, [0])
    # val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)

    # Model and optimizer
    model = SOC_Mamba_V3(d_model=d_model, n_layers=n_layers, soc_window=soc_window).to(device)
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
        if validation == 'false':
            model.train()
            total_loss = 0
            print(f"\n> Epoch {epoch+1}/{start_epoch+extra_epochs}")
            t1_start = perf_counter()

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+extra_epochs}", leave=False)
            for xb, yb in progress_bar:
                xb, yb = xb.to(device), yb.to(device)
                
                if teacher_forcing == 'true':
                    B, T, _ = yb.shape #yb.shape is (B, T, 2)
                    soc_window_size = model.soc_window
                    
                    IU = xb[:, soc_window_size:, :2]  # (B, T - soc_window, 2)
                    soc_truth = yb[:, :, 0]  # (B, T - soc_window, 1)
                    soh_truth = yb[:, :, 1]  # (B, T - soc_window, 1)

                    soc_full = soc_truth.unfold(dimension=1, size=soc_window_size, step=1)  
                    soc_full = soc_full[:, :-1, :]  
                    soc_full = soc_full + 5*noise_std * torch.randn_like(soc_full)
                    soh_full = soh_truth.unfold(dimension=1, size=soc_window_size, step=1)
                    soh_full = soh_full[:, :-1, :]  
                    soh_full = soh_full + noise_std * torch.randn_like(soh_full)

                    x_full = torch.cat([IU, soc_full, soh_full], dim=-1)  # (B, T - soc_window, 2 + 2*soc_window)

                    pred_soc, pred_soh = model(x_full)

                    loss_soc = F.mse_loss(pred_soc, soc_truth[:, soc_window_size:])
                    loss_soh = F.mse_loss(pred_soh, soh_truth[:, soc_window_size:])
                    smoothness_penalty_soc = F.mse_loss(pred_soc[:, 1:], pred_soc[:, :-1])
                    loss = loss_soc + loss_soh + lambda_smooth * (smoothness_penalty_soc)

                    # # Add debug prints or breakpoint here
                    # print(f"Loss: {loss.item():.4f}")
                    # print(f"x_full[0]: {x_full[30]}")
                    # print(f"pred_soc[0]: {pred_soc[30]}")
                    # print(f"soc_truth[0]: {soc_truth[30]}")
                    # print(f"pred_soh[0]: {pred_soh[30]}")
                    # print(f"soh_truth[0]: {soh_truth[30]}")

                    # sys.exit()
                    


                elif teacher_forcing == 'partial':
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
            
            avg_train_loss = total_loss / 300
            print(f"> Epoch time: {t1_stop - t1_start:.2f}s | Avg Train Loss: {avg_train_loss:.5f} | Train Loss: {total_loss:.5f}")

        if validation == 'true':
            # Validation (autoregressive, no teacher forcing)
            model.eval()
            val_loss_soc = 0
            val_loss_soh = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    B, T, _ = yb.shape
                    soc_window_size = model.soc_window
                    pred_seq_soc = []
                    pred_seq_soh = []
                    soc_buffer = yb[:, :soc_window_size, 0].detach().clone()
                    soc_buffer = soc_buffer + 5*noise_std * torch.randn_like(soc_buffer)
                    soh_buffer = yb[:, :soc_window_size, 1].detach().clone()
                    soh_buffer = soh_buffer + noise_std * torch.randn_like(soh_buffer)

                    for t in range(soc_window_size, T):
                        xt = xb[:, t, :2]
                        xt_input = torch.cat([xt, soc_buffer, soh_buffer], dim=1).unsqueeze(1)
                        soc_t, soh_t = model.step(xt_input)                       
                        # soc_t = soc_t.view(1)
                        # soh_t = soh_t.view(1)
                        # print(f"last soc and pred soc: {soc_buffer[:,-1], soc_t}")
                        # print(f"last soh and pred soh: {soh_buffer[:,-1], soh_t}")
                        # print(f"true soc and true soh: {yb[:,t,0], yb[:,t,1]}")
                        pred_seq_soc.append(soc_t)
                        pred_seq_soh.append(soh_t)
                        
                        soc_buffer = torch.cat([soc_buffer[:, 1:], soc_t.unsqueeze(1)], dim=1)
                        soh_buffer = torch.cat([soh_buffer[:, 1:], soh_t.unsqueeze(1)], dim=1)

                    pred_soc = torch.stack(pred_seq_soh, dim=1)  # (B, T - soc_window)
                    pred_soh = torch.stack(pred_seq_soh, dim=1)  # (B, T - soc_window)
                    instance_idx = 2  # 16th instance in batch

                    # Extract one value from each timestep
                    pred_soc_16 = torch.stack([t[instance_idx] for t in pred_seq_soc])  # shape: (295,)
                    pred_soh_16 = torch.stack([t[instance_idx] for t in pred_seq_soh])
                    true_soc_16 = yb[instance_idx, soc_window_size:, 0]
                    true_soh_16 = yb[instance_idx, soc_window_size:, 1]
                    print(f"pred_seq_soc: {pred_soc_16}")
                    print(f"true soc: {true_soc_16}")
                    print(f"pred_seq_soh: {pred_soh_16}")
                    print(f"true soh: {true_soh_16}")
                    val_loss_soc += F.mse_loss(pred_soc, yb[:, soc_window_size:,0], reduction='sum').item()
                    val_loss_soh += F.mse_loss(pred_soh, yb[:, soc_window_size:,1], reduction='sum').item()
                    sys.exit()

            avg_val_loss_soc = val_loss_soc / (len(val_set) * (y_true.size(1) - soc_window))
            avg_val_loss_soh = val_loss_soh / (len(val_set) * (y_true.size(1) - soc_window))
            # avg_val_loss_soc = val_loss_soc / (y_true.size(1) - soc_window)
            # avg_val_loss_soh = val_loss_soh / (y_true.size(1) - soc_window)
            # print(f"pred_seq_soc: {pred_seq_soc[0]}")
            # print(f"true soc: {yb[0, soc_window_size:, 0]}")
            print(f"> Validation SOC RMSE: {avg_val_loss_soc ** 0.5:.5f}")
            print(f"> Validation SOH RMSE: {avg_val_loss_soh ** 0.5:.5f}")

        scheduler.step(total_loss)

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