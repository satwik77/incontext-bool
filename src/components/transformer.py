import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import numpy as np

from src.components.transformer_utils import LearnablePositionalEncoding
from src.components.transformer_encoder import Transformer




class TransformerCLF(nn.Module):
	def __init__(self, n_dims, d_model, n_layer, n_head, dropout=0.0, pos_encode_type ='learnable'):
		super(TransformerCLF, self).__init__()
		self.model_type = 'SAN'
	
		self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)


		self.name = f"mysan_d_model={d_model}_layer={n_layer}_head={n_head}"
		self.pos_encode = True
	
		self.d_model = d_model
		self.n_dims = n_dims
		d_ffn = 4*d_model
		
		self._read_in = nn.Linear(n_dims, d_model)
		self._backbone= Transformer(d_model=d_model, n_layers=n_layer, num_heads= n_head, d_ffn=d_ffn)
		self._read_out = nn.Linear(d_model, 1)

		print('My TransformerCLF Normal Training: All parameters are tunable')

	# def init_weights(self):
	# 	initrange = 0.1
	# 	self._read_in.weight.data.uniform_(-initrange, initrange)
	# 	# if sels:
	# 	# 	self.decode()
	# 	self._read_out.weight.data.uniform_(-initrange, initrange)
	
	@staticmethod
	def _combine(xs_b, ys_b):
		"""Interleaves the x's and the y's into a single sequence."""
		bsize, points, dim = xs_b.shape
		ys_b_wide = torch.cat(
			(
				ys_b.view(bsize, points, 1),
				torch.zeros(bsize, points, dim - 1, device=ys_b.device),
			),
			axis=2,
		)
		zs = torch.stack((xs_b, ys_b_wide), dim=2)
		zs = zs.view(bsize, 2 * points, dim)
		return zs	

	def forward(self, xs, ys, inds=None):
		# input shape (xs): (batch_size, n_points, n_dims) [B L H]

		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)
		embeds = self._read_in(zs)
		embeds = embeds * math.sqrt(self.d_model)
		if self.pos_encode:
			embeds= self.pos_encoder(embeds)
		# embeds shape: (batch_size, seq_len, d_model)

		output = self._backbone(embeds)
		prediction = self._read_out(output)
		return prediction[:, ::2, 0][:, inds]  # predict only on xs







	