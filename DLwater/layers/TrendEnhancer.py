import torch
import torch.nn as nn
import torch.nn.functional as F

class TrendEnhancer(nn.Module):
    def __init__(self, channel):
        super(TrendEnhancer, self).__init__()
        self.conv_residual = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.GELU(),
            nn.Conv1d(channel, channel, kernel_size=1, bias=False)
        )
        nn.init.constant_(self.conv_residual[0].weight, 1e-3)
        nn.init.constant_(self.conv_residual[2].weight, 1e-3)

    def forward(self, x):
        """
        x: [B, C, L]
        """
        residual = self.conv_residual(x)
        return x + residual  
