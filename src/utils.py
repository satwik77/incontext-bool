import numpy as np
import torch
import ipdb as pdb
import math
from torch import linalg as LA









########## Distance from Initialization #################

def model_dist(curr_model, init_model, pos=False, weight_only=False):
    dist= 0.0
    keys = list(curr_model.state_dict().keys())
    if not pos:
        keys = [key for key in curr_model.state_dict().keys() if 'pos_encoder' not in key]
    
    if weight_only:
        keys = [key for key in curr_model.state_dict().keys() if 'weight' in key or 'bias' in key]
    
    init_params= init_model.state_dict()
    curr_params= curr_model.state_dict()
    
    for key in keys:
        assert key in init_model.state_dict().keys()
        try:
            if 'float' in str(curr_params[key].dtype):
                x = curr_params[key] - init_params[key]
        except:
            pdb.set_trace()
        
        if len(x.size())>1:
            if len(x.size())>2:
                x = x.squeeze()
                assert len(x.size())==2
                
            norm = LA.matrix_norm(x, ord='fro')
        else:
            norm = LA.vector_norm(x, ord=2)
        
        # print(key, '\t', norm)

        dist += norm**2
    
    dist= math.sqrt(dist)
    return dist




def model_sim(curr_model, init_model):
    sim_dict = {}
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    keys = list(curr_model.state_dict().keys())
    keys = [key for key in curr_model.state_dict().keys() if 'pos_encoder' not in key]
    
    

    keys = [key for key in keys if 'weight' in key]
    keys = [key for key in keys if 'attn' in key or 'mlp' in key]
    
    init_params= init_model.state_dict()
    curr_params= curr_model.state_dict()
    
    for key in keys:
        assert key in init_model.state_dict().keys()
        try:
            if 'float' in str(curr_params[key].dtype):
                # Cosine similarity between two vectors
                flat_x = curr_params[key].flatten()
                flat_y = init_params[key].flatten()
                sim = cos(flat_x, flat_y)
                sim_dict[key] = sim
        except:
            pdb.set_trace()
        
        
        
    return sim_dict