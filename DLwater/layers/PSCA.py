import torch, torch.nn as nn, torch.nn.functional as F
class Config:
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        self.n_embd = n_embd
        self.n_head = n_head
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size
class PB_RPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.d = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(self.d, self.d, bias=cfg.bias)
        self.c_proj = nn.Linear(self.d, self.d, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.gamma = nn.Parameter(torch.ones(self.nh, 1))   # 初值 1
        self.temp  = nn.Parameter(torch.ones(self.nh, 1))
    def forward(self, x, phi):
        B, T, d = x.shape
        w = self.c_attn(x).view(B, T, self.nh, d // self.nh).transpose(1, 2) 
        w_sq = w ** 2
        denom = torch.cumsum(w_sq, dim=-2).clamp_min(1e-12)
        w_norm = w_sq / denom
        tssa_score = w_norm.sum(-1) * self.temp            
        phi_Q = phi.unsqueeze(1)                          
        cum_phi  = torch.cumsum(phi,  dim=-1).unsqueeze(1) 
        steps    = torch.arange(1, T + 1, device=x.device)
        mean_phi = cum_phi / steps.view(1, 1, -1)          
        penalty  = (phi_Q - mean_phi).pow(2)              
        score = tssa_score - self.gamma * penalty          
        alpha = F.softmax(score, dim=1)
        dots  = torch.cumsum(w_sq * alpha.unsqueeze(-1), dim=-2) / \
                (alpha.cumsum(dim=-1).unsqueeze(-1) + 1e-8)
        attn  = (1.0 / (1.0 + dots)).clamp_max(1e4)
        y     = - w * alpha.unsqueeze(-1) * attn
        y     = y.transpose(1, 2).contiguous().view(B, T, d)
        y     = self.dropout(self.c_proj(y))
        return y, alpha
