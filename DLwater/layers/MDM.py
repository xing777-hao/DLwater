import torch
import torch.nn as nn
import torch.nn.functional as F

class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        L, C = input_shape           
        self.seq_len = L
        self.channels = C
        self.k = k
        self.c = c
        self.layernorm = layernorm
        self.scales = [c ** i for i in range(k, 0, -1)] + [1]

        if self.layernorm:
            self.norm = nn.BatchNorm1d(C * L)

        self.downsamples = nn.ModuleDict()
        
        for s in self.scales:
            if s > 1:
                self.downsamples[str(s)] = nn.Conv1d(
                    C, C,
                    kernel_size=2*s,
                    stride=s,
                    padding=s//2,
                    groups=C,
                    bias=False
                )
        self.projs = nn.ModuleDict()
        for s in self.scales:
            Ls = (L + s - 1) // s
            if s > 1:
                self.projs[str(s)] = nn.Sequential(
                    nn.Linear(Ls, L),
                    nn.GELU(),
                    nn.Linear(L, L)
                )
            else:
                self.projs[str(s)] = nn.Identity()

        self.gates = nn.ModuleDict()
        for s in self.scales:
            self.gates[str(s)] = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # -> [B, C, 1]
                nn.Flatten(start_dim=1),  # -> [B, C]
                nn.Linear(C, 1),
                nn.Sigmoid()
            )

    def forward(self, x):

        B, C, L = x.shape
        assert (L, C) == (self.seq_len, self.channels)

        if self.layernorm:
            x = self.norm(x.flatten(1)).view(B, C, L)

        feats = []
        for s in self.scales:
            if s > 1:
                down = self.downsamples[str(s)](x)    
                Bc, Cs, Ls = down.shape[0]*down.shape[1], down.shape[1], down.shape[2]
                down_flat = down.view(B*C, Ls)          
                up_flat = self.projs[str(s)](down_flat) 
                up = up_flat.view(B, C, self.seq_len)   
            else:
                up = x 
            feats.append(up)


        scale_tensor = torch.stack(feats, dim=1)
        num_scales = scale_tensor.shape[1]

        gate_vals = []
        for i, s in enumerate(self.scales):
            g = self.gates[str(s)](feats[i])         
            gate_vals.append(g)
        gate_tensor = torch.cat(gate_vals, dim=1)    
        gate_tensor = gate_tensor / (gate_tensor.sum(dim=1, keepdim=True) + 1e-6)
        gate_tensor = gate_tensor.view(B, num_scales, 1, 1)
        fused = (scale_tensor * gate_tensor).sum(dim=1)  
        return fused