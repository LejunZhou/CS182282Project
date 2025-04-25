import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mambapy.mamba import Mamba, MambaConfig
from soc_mamba_nar_training import MambaSOCModel

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model checkpoint
def load_model(path="saves/model.pth"):
    model = MambaSOCModel().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

# Full-sequence inference function
def full_sequence_inference(model, sequence):
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, L, 2)
        pred = model(x)  # shape: (1, L)
        return pred.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    # Load a test sequence from file (or create synthetic one)
    with open("data/90_ SOH/battery_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    test_sequence = data['X'][100]  # shape: (300, 2)
    true_soc = data['y'][100]       # shape: (300,)

    model = load_model()
    soc_preds = full_sequence_inference(model, test_sequence)

    # Plot SOC vs time step
    os.makedirs("result", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(soc_preds, label='Predicted SOC')
    plt.plot(true_soc, label='Ground Truth SOC', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("SOC")
    plt.title("Predicted vs Ground Truth SOC over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/soc_prediction.png")
    plt.close()

    print("Plot saved to result/soc_prediction.png")