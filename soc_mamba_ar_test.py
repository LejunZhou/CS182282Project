import torch
import pickle
import matplotlib.pyplot as plt
from models import SOC_Mamba_V3

def evaluate_one_instance():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load combined dataset
    with open("data/battery_dataset_combined.pkl", "rb") as f:
        X_raw, y_true = pickle.load(f)

    # Select one instance from the test portion
    n = 0  # change this to try other instances
    xb = X_raw[n:n+1].to(device)  # shape (1, T, 2)
    yb = y_true[n:n+1].to(device)  # shape (1, T, 2)

    B, T, _ = yb.shape
    model = SOC_Mamba_V3(d_model=32, n_layers=2, soc_window=5).to(device)
    model_path = "saves_ar/soc_mamba_v3/model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    soc_window = model.soc_window
    pred_seq_soc, pred_seq_soh = [], []

    soc_buffer = yb[:, :soc_window, 0].detach().clone()
    soh_buffer = yb[:, :soc_window, 1].detach().clone()

    with torch.no_grad():
        for t in range(soc_window, T):
            xt = xb[:, t, :2]  # (1, 2)
            xt_input = torch.cat([xt, soc_buffer, soh_buffer], dim=1).unsqueeze(1)  # (1, 1, 2 + 2*window)
            soc_t, soh_t = model.step(xt_input)

            pred_seq_soc.append(soc_t.detach())
            pred_seq_soh.append(soh_t.detach())

            soc_buffer = torch.cat([soc_buffer[:, 1:], soc_t.view(1, 1)], dim=1)
            soh_buffer = torch.cat([soh_buffer[:, 1:], soh_t.view(1, 1)], dim=1)

    # Stack predictions and convert to numpy for plotting
    pred_seq_soc = torch.stack(pred_seq_soc).cpu().numpy().squeeze()
    pred_seq_soh = torch.stack(pred_seq_soh).cpu().numpy().squeeze()
    gt_soc = yb[0, soc_window:, 0].cpu().numpy()
    gt_soh = yb[0, soc_window:, 1].cpu().numpy()

    t_range = list(range(soc_window, T))

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_range, gt_soc, label="True SOC", linewidth=2)
    plt.plot(t_range, pred_seq_soc, label="Predicted SOC", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("SOC (%)")
    plt.title("SOC Prediction")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_range, gt_soh, label="True SOH", linewidth=2)
    plt.plot(t_range, pred_seq_soh, label="Predicted SOH", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("SOH (%)")
    plt.title("SOH Prediction")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("soc_soh_test_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate_one_instance()