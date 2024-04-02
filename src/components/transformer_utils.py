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


def self_attention(query, key, value, mask):
    '''
    Args:
        query: Tensor of shape [batch, seq, num_heads, emb_dim / num_heads].
        value: Tensor of shape [batch, seq, num_heads, emb_dim / num_heads].
        key: Tensor of shape [batch, seq, num_heads, emb_dim / num_heads].
        mask: Tensor of shape [1, 1, seq, seq]
    '''
    
    head_dim = query.size(-1)
    
    attention_logits = torch.einsum("bthd, bThd -> bhtT", query, key)  # [B N L L]
    attention_logits /= math.sqrt(head_dim)
    
    if mask is not None:
        min_value = torch.finfo(attention_logits.dtype).min
        attention_logits = attention_logits.masked_fill(mask ==0, min_value)
    
    
    attention_weights = F.softmax(attention_logits, -1)     # [B N L L]
    
    attention_vec = torch.einsum("bhtT,bThd->bthd", attention_weights, value)
    return attention_vec
    


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
