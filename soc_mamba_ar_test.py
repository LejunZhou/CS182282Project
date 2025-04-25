import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from mambapy.mamba import Mamba, MambaConfig, RMSNorm

# ------------------ Model ------------------
from models import *
    
# ------------------ Load and Evaluate ------------------
def evaluate_one_instance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("data/90_ SOH/battery_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    X_raw = torch.tensor(data['X'], dtype=torch.float32)
    y_true = torch.tensor(data['y'], dtype=torch.float32)

    model = SOC_LM(d_model=32, n_layers=2, soc_window=4).to(device)
    checkpoint = torch.load("saves_ar/soc_lm/model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    n = 0
    xb = X_raw[n+0:n+1].to(device)  # shape (1, T, 2)
    yb = y_true[n+0:n+1].to(device)  # shape (1, T)
    B, T = yb.shape
    w = model.soc_window

    # Seed buffer with ground truth
    soc_buffer = yb[:, :w].detach().clone()  # (1, w)
    pred_seq = []

    for t in range(w, T):
        xt = xb[:, t, :2]  # (1, 2)
        xt_input = torch.cat([xt, soc_buffer], dim=1).unsqueeze(1)  # (1, 1, 2 + w)
        soc_t = model.step(xt_input)
        pred_seq.append(soc_t.item())
        soc_buffer = torch.cat([soc_buffer[:, 1:], soc_t.unsqueeze(1)], dim=1)

    pred_seq_plot = []
    for i in pred_seq:
        pred_seq_plot.append(i*100)
    # Plot results
    t_range = list(range(w, T))
    plt.figure(figsize=(10, 4))
    plt.plot(t_range, 100*yb.squeeze(0).cpu().numpy()[w:], label="Ground Truth", linewidth=2)
    plt.plot(t_range, pred_seq_plot, label="Prediction", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("SOC(%)")
    plt.title("Mamba SOC Prediction (One Instance)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("soc_prediction_plot.png")
    plt.show()

if __name__ == "__main__":
    evaluate_one_instance()