import torch
import torch.nn as nn
import math
from layers.network import Network
from layers.revin import RevIN
from layers.MDM import MDM
from layers.HASD import hasd
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = configs.seq_len   
        pred_len = configs.pred_len 
        c_in = configs.enc_in       
        self.seq_len  = configs.seq_len  
        self.pred_len = configs.pred_len 
        enc_in   = configs.enc_in
        cycle_len = configs.cycle_len
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)
        self.ma_type = configs.ma_type
        alpha = configs.alpha      
        beta = configs.beta        
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch,enc_in,cycle_len)
        self.mdm = MDM((seq_len,enc_in), k=3, c=2, layernorm=True).to(self.device)
        self.mamsd = hasd(input_shape=(seq_len, enc_in),alpha_init=alpha,k=3, c=2,layernorm=True).to(self.device)
    def forward(self, x, cycle_index):
        if self.revin:
            x = self.revin_layer(x, 'norm')          


        x_c = x.permute(0, 2, 1)                      

        if self.ma_type == 'reg':
            season_out = x_c                          
            trend_out  = torch.zeros_like(x_c)        
        else:
            season_out, trend_out = self.mamsd(x_c)   

        season_out = season_out.permute(0, 2, 1)      
        trend_out  = trend_out.permute(0, 2, 1)       

        x = self.net(season_out,trend_out, cycle_index)

        if self.revin:
            x = self.revin_layer(x, 'denorm')        

        return x
