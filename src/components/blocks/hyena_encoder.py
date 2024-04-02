import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import numpy as np

from src.components.blocks.hyena_utils import HyenaOperator


class HyenaBlock(nn.Module):
    "Encoder is made up of retention and feed forward (defined below)"

    def __init__(self, d_model, order=3, d_ffn=0, dropout=0.0):
        super(HyenaBlock, self).__init__()
        self.d_model = d_model
        self.order = order
        
        if d_ffn == 0:
            self.d_ffn = 4*self.d_model
        else:
            self.d_ffn = d_ffn

        self.ffn =  nn.Sequential(nn.Linear(self.d_model, self.d_ffn), nn.GELU(), nn.Linear(self.d_ffn, self.d_model) )
        
        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)

        self.hyena_operator = HyenaOperator(d_model=self.d_model,  order=self.order)
        
        self.drop = nn.Dropout(p=dropout)

        
    
    def forward(self, x):
        '''
            x: [B L H]
            decay: [N L L]
        '''
        
        ln_x = self.ln_1(x)                     # Layernorm

        y = self.hyena_operator(ln_x)           # Hyena Op [B L H] -> [B L H] 

        x = x + y                               # Residual connection
        ln_x2 = self.ln_2(x)                    # Layernorm 
    
        out = self.ffn(ln_x2)                   # FFN [B L H] -> [B L H]
        x = x + out                             # Residual connection
        
        return x
        
    




class Hyena(nn.Module):
    "Stack of N Hyena Blocks"

    def __init__(self, d_model, n_layers, order, d_ffn=0, dropout=0.0):
        super(Hyena, self).__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.order = order
        self.layernorm = nn.LayerNorm(d_model)


        
        self.blocks = nn.ModuleList([HyenaBlock(d_model, order, d_ffn, dropout) for _ in range(n_layers)])
        


        # self.initialize(init_range = 0.02)
    


    def initialize(self, init_range=0.02):
        for key in self.state_dict().keys():
            if 'weight' in key:
                self.state_dict()[key].data.normal_(mean=0.0, std= init_range)
            elif 'bias' in key:
                self.state_dict()[key].data.zero_()
            
            if 'ffn.2.weight' in key or 'attn_o.weight' in key:
                self.state_dict()[key].data.normal_(mean=0.0, std= init_range/math.sqrt(2 * self.n_layers))
            
            if 'ln_' in key and 'weight' in key:
                self.state_dict()[key].data.fill_(1.0)
            if 'ln_' in key and 'bias' in key:
                self.state_dict()[key].data.zero_()


    def forward(self, x):
        '''
            Pass the input through each layer in turn.
            x: [B L H]
        '''
        for blocks in self.blocks:
            x = blocks(x)
        return self.layernorm(x)


