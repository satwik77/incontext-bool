import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import numpy as np

from src.components.transformer_utils import clones, head_split, self_attention


class TransformerBlock(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, num_heads, d_ffn=0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads= num_heads
        self.head_dim = d_model // num_heads
        
        if d_ffn == 0:
            self.d_ffn = 4*self.d_model
        else:
            self.d_ffn = d_ffn

        self.ffn =  nn.Sequential(nn.Linear(self.d_model, self.d_ffn), nn.GELU(), nn.Linear(self.d_ffn, self.d_model) )
        
        self.ln_1 = nn.LayerNorm(self.d_model)
        self.ln_2 = nn.LayerNorm(self.d_model)

        self.proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.attn_o = nn.Linear(self.d_model, self.d_model)
        
        self.drop = nn.Dropout(p=dropout)
        
    
    def forward(self, x, mask):
        '''
            x: [B L H]
        '''
        
        ln_x = self.ln_1(x)
        
        ### Attention Block ###
        projections = self.proj(ln_x)               # [B L H] -> [B L 3*H]
        projections = projections.unsqueeze(-1)  # [B L 3H 1]
        q, k, v= torch.split(projections, self.d_model, dim=-2)   # [B L H 1] x 3
        
        q = head_split(q, self.head_dim)      # [B L N D] 
        k = head_split(k, self.head_dim)      # [B L N D]
        v = head_split(v, self.head_dim)      # [B L N D]
        
        attn_out= self_attention(query = q, key = k, value = v, mask = mask)
        attn_out = attn_out.reshape(*x.shape[:2], self.d_model)      # [B L N D] -> [B L H]
        attn_out = self.attn_o(attn_out)
        
        ### Residual + Layernorm ###
        x = x + attn_out
        ln_x2 = self.ln_2(x)
        
        ### FFN Block ###
        out = self.ffn(ln_x2)
        
        x = x + out
        
        return x
        
    




class Transformer(nn.Module):
    "Stack of N Transformer Blocks"

    def __init__(self, d_model, n_layers, num_heads, d_ffn=0, dropout=0.0):
        super(Transformer, self).__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        transformer_block = TransformerBlock(d_model, num_heads, d_ffn, dropout)
        self.blocks = clones(transformer_block, n_layers)
        self.layernorm = nn.LayerNorm(d_model)
        # self.initialize(init_range = 0.02)
    
    def get_mask(self, L, device):
        mask = torch.tril(torch.ones(L,L))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask.to(device)
    
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
            Pass the input (and mask) through each layer in turn.
            x: [B L H]
        '''
        mask = self.get_mask(x.size(1), x.device)
        for blocks in self.blocks:
            x = blocks(x, mask)
        return self.layernorm(x)



# class Encoder(nn.Module):
#     "Core encoder is a stack of N layers"

#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.d_model)

#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)


# class EncoderLayer(nn.Module):
#     "Encoder is made up of self-attn"

#     def __init__(self, d_model, self_attn, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.d_model = d_model
#         self.sublayer = clones(SublayerConnection(d_model, dropout), 1)
#         #self.feed_forward = feed_forward

#     def forward(self, x, mask):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return x

# class EncoderLayerFFN(nn.Module):
#     "Encoder is made up of self-attn and feed forward (defined below)"

#     def __init__(self, d_model, self_attn, feed_forward, dropout):
#         super(EncoderLayerFFN, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
#         self.d_model = d_model


#     def forward(self, x, mask):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)
#         # return self.feed_forward(self.self_attn(x, x, x, mask))
