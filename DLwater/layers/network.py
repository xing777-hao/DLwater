import torch
from torch import nn
import math
import torch.nn.functional as F
from layers.PSCA import PB_RPA,Config
from layers.TrendEnhancer import TrendEnhancer

def hermitian_attn(Qr, Qi, K, V):
    S = torch.einsum('bcl,bck->blk', Qr, K)   
    S = S / math.sqrt(Qr.size(1))
    A = torch.softmax(S, dim=-1)
    context = torch.einsum('blk,bck->bcl', A, V)   
    return context

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,enc_in,cycle_len,top_m=3):
        super(Network, self).__init__()
        self.top_m = top_m
        if isinstance(cycle_len, int):       
            cycle_len = [cycle_len]
        self.cycle_lens = cycle_len
        self.num_scales = len(cycle_len)
        self.enc_in = enc_in
        self.pred_len = pred_len

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': 
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1


        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

 
        self.fc2 = nn.Linear(self.dim, patch_len)

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        self.fc7 = nn.Linear(pred_len // 2, pred_len)
        self.fc8 = nn.Linear(pred_len * 2, pred_len)
        self.beta = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        self.temporalQuery_list = nn.ParameterList([
            nn.Parameter(torch.zeros(Wk, enc_in), requires_grad=True)
            for Wk in self.cycle_lens
        ])
        self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)
        if isinstance(cycle_len, int):
            cycle_len = [cycle_len]
        self.cycle_lens = cycle_len
        self.num_scales = len(cycle_len)
        self.A_list   = nn.ParameterList([
            nn.Parameter(torch.randn(Wk, enc_in)) for Wk in self.cycle_lens
        ])
        self.phi_list = nn.ParameterList([
            nn.Parameter(2*math.pi*torch.rand(Wk, enc_in)) for Wk in self.cycle_lens
        ])
        self.beta = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        self.global_beta = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),        
            nn.Flatten(1),                  
            nn.Linear(enc_in, enc_in // 2),
            nn.GELU(),
            nn.Linear(enc_in // 2, self.num_scales)   
        )
        cfg = Config(
            n_embd=patch_len,    
            n_head=8,
            bias=True,
            dropout=0.1,
            block_size=self.patch_num  
        )
        self.trend_enhancer = TrendEnhancer(channel=enc_in)  
        self.psca = PB_RPA(cfg)  
    def forward(self, s, t,cycle_index):

        s = s.permute(0, 2, 1)   
        t = t.permute(0, 2, 1)
        B, C, I = s.shape
        device  = s.device
        cycle_index = cycle_index.to(device)

        p_logits = self.router(s)               
        p_soft   = torch.softmax(p_logits, dim=-1)
        p_weights = torch.softmax(self.router(s), dim=-1)
        c_list = []
        phi_list=[]
        for k, (A_tab, P_tab, W_k) in enumerate(zip(self.A_list, self.phi_list, self.cycle_lens)):
            idx  = (cycle_index[:, None] + torch.arange(I, device=device)) % W_k
            A    = F.softplus(A_tab[idx]).permute(0, 2, 1)
            phi  = P_tab[idx].permute(0, 2, 1)
            Qr   = A * torch.cos(phi)
            Qi   = A * torch.sin(phi)
            c_k  = hermitian_attn(Qr, Qi, s, s)       
            c_list.append(c_k)
            phi_list.append(phi)
        c_stack = torch.stack(c_list, dim=1)            
        phi_stack = torch.stack(phi_list, dim=1)        

        beta = p_weights.unsqueeze(-1).unsqueeze(-1)            
        channel_information = (beta * c_stack).sum(dim=1)       
        fused_phi = (beta * phi_stack).sum(dim=1)
        s = s + channel_information
        s = torch.reshape(s, (B*C, I)) 
        t = torch.reshape(t, (B*C, I)) 
        phi = fused_phi.mean(1)
        phi = phi.repeat_interleave(C, dim=0)
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
            phi = F.pad(phi, (0, self.stride), mode='replicate')
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        phi = phi.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        phi_scalar = phi.mean(-1) 
        res = s
        s,alpha=self.psca(s,phi_scalar)
        s = s + res
        s = self.gelu2(s)
        s = self.bn2(s)
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)                     
        t = t.view(B, C, -1)  
        trend_fft = torch.fft.rfft(t, dim=-1)
        low_freq_mask = torch.zeros_like(trend_fft)
        low_freq_mask[..., :5] = 1  
        trend_lowfreq = trend_fft * low_freq_mask
        trend_enhanced = torch.fft.irfft(trend_lowfreq, n=t.size(-1), dim=-1)
        t = t + trend_enhanced  
        t = self.trend_enhancer(t)  
        t = t.view(B*C, -1)  
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0,2,1)
        return x
