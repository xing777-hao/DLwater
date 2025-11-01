import torch
import torch.nn as nn
import torch.nn.functional as F
def ema_vec(x_ds, alpha_bc):
    _, _, Ls = x_ds.shape
    device = x_ds.device
    powers = torch.arange(Ls-1, -1, -1, dtype=torch.double, device=device)
    base   = torch.pow(1 - alpha_bc.double(), powers)         
    w_num  = base.clone()
    w_num[..., 1:] *= alpha_bc
    num = torch.cumsum(x_ds * w_num.float(), dim=-1)

    den = torch.cumsum(base.float(), dim=-1).clamp_min(1e-12)  
    return num / den                                           



class hasd(nn.Module):
    def __init__(self,input_shape,alpha_init=0.3,k=3, c=2,
                 layernorm=True):
        super().__init__()
        L, C = input_shape
        self.seq_len, self.channels = L, C
        self.layernorm = layernorm
        self.scales = [c ** i for i in range(k, 0, -1)] + [1]

        if layernorm:
            self.norm = nn.BatchNorm1d(C * L)

        self.downsamples = nn.ModuleDict()
        self.projs       = nn.ModuleDict()
        self.gates       = nn.ModuleDict()
        self.alpha_tables= nn.ParameterDict()         

        for s in self.scales:
            Ls = (L + s - 1) // s
            if s > 1:
                self.downsamples[str(s)] = nn.Conv1d(
                    C, C, kernel_size=2 * s, stride=s,
                    padding=s // 2, groups=C, bias=False
                )
                self.projs[str(s)] = nn.Sequential(
                    nn.Linear(Ls, L), nn.GELU(), nn.Linear(L, L)
                )
            else:                                     
                self.projs[str(s)] = nn.Identity()

            self.gates[str(s)] = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(C, 1),
                nn.Sigmoid()
            )
            self.alpha_tables[str(s)] = nn.Parameter(
                torch.full((1, C, 1), alpha_init, dtype=torch.float32)
            )
    def forward(self, x):                    
        B, C, L = x.shape
        if self.layernorm:
            x = self.norm(x.flatten(1)).view(B, C, L)

        trend_feats, season_feats = [], []
        for s in self.scales:
            if s > 1:
                down = self.downsamples[str(s)](x)      
            else:
                down = x
            B_, C_, Ls = down.size()
            if s > 1:
                up_flat = self.projs[str(s)](
                    down.view(B_ * C_, Ls)
                )                                        
                up = up_flat.view(B_, C_, L)             
            else:
                up = down                               
            alpha_bc = torch.sigmoid(self.alpha_tables[str(s)])  
            trend_ds = ema_vec(down, alpha_bc)          

            if s > 1:
                trend_up = self.projs[str(s)](
                    trend_ds.view(B_ * C_, Ls)
                ).view(B_, C_, L)                         
            else:
                trend_up = trend_ds                      

            season_up = up - trend_up                     

            trend_feats.append(trend_up)
            season_feats.append(season_up)
        gate_vals = [self.gates[str(s)](season_feats[i])
                     for i, s in enumerate(self.scales)]  
        gate_tensor = torch.cat(gate_vals, dim=1)        
        gate_tensor = gate_tensor / (gate_tensor.sum(dim=1, keepdim=True) + 1e-6)
        gate_tensor = gate_tensor.view(B, len(self.scales), 1, 1)
        trend_stack  = torch.stack(trend_feats,  dim=1)    
        season_stack = torch.stack(season_feats, dim=1)  

        trend_out  = (trend_stack  * gate_tensor).sum(dim=1)   
        season_out = (season_stack * gate_tensor).sum(dim=1)  
        return season_out, trend_out
