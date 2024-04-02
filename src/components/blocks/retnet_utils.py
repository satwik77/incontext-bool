import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import numpy as np
import copy


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def head_split(x, head_dim: int):
    x = x.reshape(*x.shape[:2], -1, head_dim)
    return x




def fixed_pos_embedding(x):
    dim = x.shape[-1]
    L = x.shape[1]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(L), inv_freq)
    return torch.tensor(np.sin(sinusoid_inp)), torch.tensor(np.cos(sinusoid_inp))


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_pos_emb(X):
    """
    X: [batch, length, num_heads, head_dim]
    """
    sin, cos = fixed_pos_embedding(X)  # [L H/2]

    sin = sin.to(X.device)
    cos = cos.to(X.device)

    # Repeat along the last dimension
    sin = sin.repeat(1, 2)[None, :, None, :]  # [1, length, 1, head_dim]
    cos = cos.repeat(1, 2)[None, :, None, :]  # [1, length, 1, head_dim]

    return (X * cos) + (rotate_every_two(X) * sin)


def retention(query, key, value, decay):
    '''
    Args:
        query: Tensor of shape [batch, seq, num_heads, emb_dim / num_heads].
        value: Tensor of shape [batch, seq, num_heads, emb_dim / num_heads].
        key: Tensor of shape [batch, seq, num_heads, emb_dim / num_heads].
        decay: Tensor of shape [1, n, seq, seq]
    '''
    
    # pdb.set_trace()
    retention_logits = torch.einsum("bthd, bThd -> bhtT", query, key)  # [B N L L]
    retention_logits = retention_logits * decay

    retention_logits = retention_logits.to(dtype=value.dtype)
    
    retention_vec = torch.einsum("bhtT,bThd->bthd", retention_logits, value)
    return retention_vec
    


class LearnablePositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=200, init_range = 0.1):
		super(LearnablePositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		# pos_embeds = torch.FloatTensor(max_len, 1, d_model).uniform_(-init_range, init_range)
		pos_embeds = torch.FloatTensor(1, max_len, d_model).uniform_(-init_range, init_range)
		pe = nn.Parameter(pos_embeds, requires_grad = True)
		self.pe = pe

	def forward(self, x):
		#pdb.set_trace()
		x = x + self.pe[:, :x.size(1), :]
		return self.dropout(x)
