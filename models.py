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
    
class SOC_Mamba_V2(nn.Module):
    def __init__(self, d_model=16, n_layers=1, soc_window=1):
        super().__init__()
        self.soc_window = soc_window
        self.config = MambaConfig(d_model=d_model, n_layers=n_layers)

        self.input_proj = nn.Linear(2 + soc_window, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(self.config)
        self.norm = RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.output_head = nn.Linear(d_model, 1)


    def _compute_delta(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.mamba(x)
        x = self.norm(x)
        delta = self.output_head(x)
        return delta

    def forward(self, x):
        """
        x: (B, T, 2 + soc_window)
        Returns: (B, T), (B, T)
        """
        past_soc = x[:,:, -1].unsqueeze(-1)  # (B, T, 1)

        delta = self._compute_delta(x)                   # (B, T, 1)
        print(f"delta: {delta.shape}")
        print(f"past_soc: {past_soc.shape}")
        y = past_soc + delta
        return y.squeeze(-1), delta.squeeze(-1)

    def step(self, x_t):
        """
        x_t: (B, 1, 2 + soc_window)
        Returns: (B,)
        """
        past_soc = x_t[:,:, -1].unsqueeze(-1)  # (B, T, 1)

        delta = self._compute_delta(x_t)                  # (B, 1, 1)
        y = past_soc + delta
        return y.squeeze(-1).squeeze(-1)
    
class SOC_Mamba_V3(nn.Module):
    def __init__(self, d_model=16, n_layers=1, soc_window=3):
        super().__init__()
        self.soc_window = soc_window
        self.config = MambaConfig(d_model=d_model, n_layers=n_layers)

        self.input_proj = nn.Linear(2 + 2*soc_window, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(self.config)
        self.norm = RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.output_head = nn.Linear(d_model, 2)


    def _compute_delta(self, x):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.mamba(x)
        x = self.norm(x)
        delta = self.output_head(x)
        delta_soc, delta_soh = delta[:,:,0], delta[:,:,1]
        return delta_soc, delta_soh

    def forward(self, x):
        """
        x: (B, T, 2 + 2*soc_window)
        Returns: (B, T), (B, T)
        """
        past_soc = x[:,:, -(1+self.soc_window)]  # (B, T, 1)
        past_soh = x[:,:, -1]  # (B, T, 1)

        delta_soc, delta_soh = self._compute_delta(x)                   # (B, T, 1)
        # normalize delta_soc to be between -0.05 and 0.05
        delta_soc = torch.clamp(delta_soc, -0.005, 0.005)
        # normalize delta_soh to be between -0.001 and 0.001
        delta_soh = torch.clamp(delta_soh, -0.0001, 0.0001)
        y_1 = past_soc + delta_soc
        y_2 = past_soh + delta_soh
        return y_1.squeeze(-1), y_2.squeeze(-1) 

    def step(self, x_t):
        """
        x_t: (B, 1, 2 + 2*soc_window)
        Returns: (B,)
        """
        past_soc = x_t[:,:, -(1+self.soc_window)]  # (B, T, 1)
        past_soh = x_t[:,:, -1]  # (B, T, 1)
        delta_soc, delta_soh = self._compute_delta(x_t)                   # (B, T, 1)
        # normalize delta_soc to be between -0.05 and 0.05
        delta_soc = torch.clamp(delta_soc, -0.005, 0.005)
        # normalize delta_soh to be between -0.001 and 0.001
        delta_soh = torch.clamp(delta_soh, -0.000001, 0.000001)
        # print(f"delta_soc: {delta_soc.shape, delta_soc}")
        # print(f"delta_soh: {delta_soh.shape, delta_soh}")
        y_1 = past_soc + delta_soc
        y_2 = past_soh + delta_soh
        return y_1.squeeze(-1).squeeze(-1), y_2.squeeze(-1).squeeze(-1)
