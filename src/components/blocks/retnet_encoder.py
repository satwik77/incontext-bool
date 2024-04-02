import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import numpy as np

from src.components.blocks.retnet_utils import head_split, retention, apply_rotary_pos_emb


class RetnetBlock(nn.Module):
	"Encoder is made up of retention and feed forward (defined below)"

	def __init__(self, d_model, num_heads, d_ffn=0, use_gate=True, dropout=0.0):
		super(RetnetBlock, self).__init__()
		self.d_model = d_model
		self.num_heads= num_heads
		self.head_dim = d_model // num_heads
		self.use_gate = use_gate
		
		if d_ffn == 0:
			self.d_ffn = 4*self.d_model
		else:
			self.d_ffn = d_ffn

		self.ffn =  nn.Sequential(nn.Linear(self.d_model, self.d_ffn), nn.GELU(), nn.Linear(self.d_ffn, self.d_model) )
		
		self.ln_1 = nn.LayerNorm(self.d_model)
		self.ln_2 = nn.LayerNorm(self.d_model)
		self.grp_norm = nn.GroupNorm(self.num_heads, self.d_model)

		self.proj = nn.Linear(self.d_model, 4 * self.d_model)
		self.ret_o = nn.Linear(self.d_model, self.d_model)
		
		self.drop = nn.Dropout(p=dropout)
		self.swish_act = nn.SiLU()
		
	
	def forward(self, x, decay):
		'''
			x: [B L H]
			decay: [N L L]
		'''
		
		ln_x = self.ln_1(x)
		
		### Retention Block ###
		projections = self.proj(ln_x)                             # [B L H] -> [B L 4*H]
		projections = projections.unsqueeze(-1)                   # [B L 4H 1]
		q, k, v, xo= torch.split(projections, self.d_model, dim=-2)   # [B L H 1] x 4
		
		q = head_split(q, self.head_dim)      # [B L N D] 
		k = head_split(k, self.head_dim)      # [B L N D]
		v = head_split(v, self.head_dim)      # [B L N D]

		# q = apply_rotary_pos_emb(q)  # [batch, seq, num_heads, head_dim]
		# k = apply_rotary_pos_emb(k)  # [batch, seq, num_heads, head_dim]
		
		ret_out= retention(query = q, key = k, value = v, decay = decay)
		ret_out = ret_out.reshape(*x.shape[:2], self.d_model)      # [B L N D] -> [B L H]
		# ret_out = ret_out.transpose(1, 2)                          # [B L H] -> [B H L]
		# ret_out = self.grp_norm(ret_out)                           # [B H L]        Group Norm expects H to be in second dimension
		# ret_out = ret_out.transpose(1, 2)                          # [B H L] -> [B L H] 

		if self.use_gate:
			xo = self.swish_act(xo.squeeze(-1))                 # [B L H 1] -> [B L H]
			ret_out = torch.einsum('blh,blh->blh', ret_out, xo) # [B L H] * [B L H] -> [B L H]
			
		ret_out = self.ret_o(ret_out)
		
		### Residual + Layernorm ###
		x = x + ret_out
		ln_x2 = self.ln_2(x)
		
		### FFN Block ###
		out = self.ffn(ln_x2)
		
		x = x + out
		
		return x
		
	




class Retnet(nn.Module):
	"Stack of N Retnet Blocks"

	def __init__(self, d_model, n_layers, num_heads, d_ffn=0, use_decay=True, use_gate=True, dropout=0.0):
		super(Retnet, self).__init__()
		
		self.n_layers = n_layers
		self.d_model = d_model
		self.n_heads = num_heads
		self.layernorm = nn.LayerNorm(d_model)
		self.use_decay = use_decay
		self.use_gate = use_gate

		# retnet_block = RetnetBlock(d_model, num_heads, d_ffn, use_gate, dropout)
		# self.blocks = clones(retnet_block, n_layers)

		# self.blocks = []
		# for _ in range(n_layers):
		#     self.blocks.append(RetnetBlock(d_model, num_heads, d_ffn, use_gate, dropout))
		
		self.blocks = nn.ModuleList([RetnetBlock(d_model, num_heads, d_ffn, use_gate, dropout) for _ in range(n_layers)])
		
		self.gamma = 1.0 - 2.0**(-5-torch.arange(self.n_heads)) 

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p) 
		# self.initialize(init_range = 0.02)
	

	def get_decay(self, L, use_decay=True, device=None):
		if use_decay:
			# D : L x L matrix with D[i, j] = gamma ** (i - j) if i >= j else 0
			n = torch.arange(L, dtype= torch.float32)[:, None]  # [L, 1]
			m = torch.arange(L, dtype= torch.float32)[None, :]  # [1, L]
			gam_expanded = self.gamma[:, None, None]  # [n_heads, 1, 1]
			D = torch.where(n >= m, gam_expanded ** (n - m), torch.zeros_like(n))  # [n_heads, L, L]
		else:
			# D is a lower triangular matrix with ones
			D = torch.tril(torch.ones((self.n_heads, L, L)))  # [n_heads, L, L]

		D = D.unsqueeze(0)      # [1, n_heads, L, L]

		return D.to(device)     # [1, n_heads, L, L]
	 

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
		decay = self.get_decay(x.size(1), True, x.device)
		# pdb.set_trace()
		for blocks in self.blocks:
			x = blocks(x, decay)
		return self.layernorm(x)


