import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from mambapy.mamba import Mamba, MambaConfig, RMSNorm


# ------------------  Models ------------------
class SOC_LM(nn.Module):
    def __init__(self, d_model=32, n_layers=2, soc_window=10):
        super().__init__()
        self.soc_window = soc_window
        self.config = MambaConfig(d_model=d_model, n_layers=n_layers)

        # Input is: [I, U, SOC_{t-1}, ..., SOC_{t-W}]
        self.input_proj = nn.Linear(2 + soc_window, d_model)
        self.mamba = Mamba(self.config)
        self.norm = RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, 2 + soc_window)
        Returns: Tensor of shape (B, T)
        """
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.mamba(x)            # (B, T, d_model)
        x = self.norm(x)             # (B, T, d_model)
        y = self.output_head(x)      # (B, T, 1)
        return y.squeeze(-1)         # (B, T)

    def step(self, x_t):
        """
        x_t: Tensor of shape (B, 1, 2 + soc_window)
        Returns: prediction for SOC_t, shape (B,)
        """
        x = self.input_proj(x_t)
        x = self.mamba(x)
        x = self.norm(x)
        y = self.output_head(x)
        return y.squeeze(-1).squeeze(-1)
    
class SOC_Mamba_V1(nn.Module):
    def __init__(self, d_model=32, n_layers=2, soc_window=10):
        super().__init__()
        self.soc_window = soc_window
        self.config = MambaConfig(d_model=d_model, n_layers=n_layers)

        self.iu_mean = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=False)
        self.iu_std = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=False)

        # Input is: [I, U, SOC_{t-1}, ..., SOC_{t-W}]
        self.input_proj = nn.Linear(2 + soc_window, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(self.config)
        self.norm = RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, 2 + soc_window)
        Returns: Tensor of shape (B, T)
        """
        # Normalize the first two input features (I, U)
        iu = x[..., :2]
        soc = x[..., 2:]
        iu_norm = (iu - self.iu_mean) / (self.iu_std + 1e-6)
        x = torch.cat([iu_norm, soc], dim=-1)

        x = self.input_proj(x)       # (B, T, d_model)
        x = self.input_norm(x)       # (B, T, d_model)
        x = self.mamba(x)            # (B, T, d_model)
        x = self.norm(x)             # (B, T, d_model)
        y = self.output_head(x)      # (B, T, 1)
        return y.squeeze(-1)         # (B, T)

    def step(self, x_t):
        """
        x_t: Tensor of shape (B, 1, 2 + soc_window)
        Returns: prediction for SOC_t, shape (B,)
        """
        # Normalize the first two input features (I, U)
        iu = x_t[..., :2]
        soc = x_t[..., 2:]
        iu_norm = (iu - self.iu_mean) / (self.iu_std + 1e-6)
        x = torch.cat([iu_norm, soc], dim=-1)

        x = self.input_proj(x_t)
        x = self.input_norm(x)
        x = self.mamba(x)
        x = self.norm(x)
        y = self.output_head(x)
        return y.squeeze(-1).squeeze(-1)
    
# ------------------ Running Normalizer ------------------
class RunningNormalizer(nn.Module):
    def __init__(self, feature_dim=2, momentum=0.01):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_var", torch.ones(feature_dim))
        self.momentum = momentum

    def forward(self, x, update=True):
        """
        x: (B, 2)
        Returns: normalized x (B, 2)
        """
        if update:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        std = torch.sqrt(self.running_var + 1e-6)
        return (x - self.running_mean) / std