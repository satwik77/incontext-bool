import torch

def prefix_scoring_step(model, xs, ys, num_repeats=4):
	model.eval()
	prefix_score = 0.0
	head_scores_dict = {}
	num_induc_heads = 0
	with torch.no_grad():
		attns = model.get_attns(xs, ys) # n_layers len of Tuple of (bs x heads x seq_len x seq_len) elements

		layer_scores = 0.0
		layer_num = 1
		for layer_attns in attns: # across layers
			head_scores = 0.0
			for zj in range(layer_attns.shape[1]): # across heads
				example_scores = 0.0
				for zi in range(layer_attns.shape[0]): # across bs examples
					attn_score = 0.0
					tots = 0
					attn_mx = layer_attns[zi][zj] # seq_len x seq_len
					n_points = int(attn_mx.shape[0]/num_repeats)
					for segment in range(1, num_repeats):
						for prev_segment in range(segment):
							for idx in range(0, n_points, 2):
								src_token = segment * n_points + idx
								trg_token = prev_segment * n_points + idx + 1
								attn_val = attn_mx[src_token][trg_token]
								attn_score += attn_val
								tots += 1
					cur_example_score = attn_score/tots
					example_scores += cur_example_score
					
				cur_head_score = example_scores/layer_attns.shape[0]
				if cur_head_score >= 0.5:
					num_induc_heads += 1
				head_scores += cur_head_score

				head_scores_dict["layer-"+str(layer_num)+"-head-"+str(zj+1)] = cur_head_score.detach().cpu().item()
				
			cur_layer_score = head_scores/layer_attns.shape[1]
			layer_scores += cur_layer_score

			layer_num += 1

		prefix_score = layer_scores/len(attns)

	model.train()
	return prefix_score.detach().cpu().item(), num_induc_heads, head_scores_dict

def nn_scoring_step(model, xs, ys, start_point, num_neighbours, device):
	model.eval()
	nn_score = 0.0
	num_nn_heads = 0
	head_scores_dict = {}
	xs_norm = torch.norm(xs, dim=2, keepdim=True)
	xs_normalized = xs / xs_norm
	xs_T = torch.transpose(xs_normalized, 1, 2) # bs x n_dims x n_points
	sim_mx = torch.matmul(xs_normalized, xs_T) # bs x n_points x n_points
	n_points = xs.shape[1]
	# data_sq_norms = torch.sum(xs**2, dim=2, keepdim=True)
	# sim_mx = torch.sqrt(torch.max(data_sq_norms + data_sq_norms.transpose(1, 2) - 2 * torch.bmm(xs, xs.transpose(1, 2)), torch.zeros(1).to(device)))
	with torch.no_grad():
		attns = model.get_attns(xs, ys) # n_layers len of Tuple of (bs x heads x seq_len x seq_len) elements ---- seq_len = 2 * n_points

		layer_scores = 0.0
		layer_num = 1
		for layer_attns in attns: # across layers
			head_scores = 0.0
			for zj in range(layer_attns.shape[1]): # across heads
				example_scores = 0.0
				for zi in range(layer_attns.shape[0]): # across bs examples
					attn_score = 0.0
					tots = 0
					attn_mx = layer_attns[zi][zj] # seq_len x seq_len
					similarities = sim_mx[zi] # n_points x n_points

					for idx in range(start_point-1, n_points, 1):
						src_token = idx*2
						if num_neighbours > idx:
							_, trg_tokens = torch.topk(similarities[idx][:idx], idx)
						else:
							_, trg_tokens = torch.topk(similarities[idx][:idx], num_neighbours)
						trg_tokens = 2*trg_tokens+1
						attn_val = torch.sum(attn_mx[src_token][trg_tokens])
						attn_score += attn_val
						tots += 1
					cur_example_score = attn_score/tots
					example_scores += cur_example_score
					
				cur_head_score = example_scores/layer_attns.shape[0]
				if cur_head_score >= 0.5:
					num_nn_heads += 1
				head_scores += cur_head_score

				head_scores_dict["layer-"+str(layer_num)+"-head-"+str(zj+1)] = cur_head_score.detach().cpu().item()
				
			cur_layer_score = head_scores/layer_attns.shape[1]
			layer_scores += cur_layer_score

			layer_num += 1

		nn_score = layer_scores/len(attns)

	model.train()
	return nn_score.detach().cpu().item(), num_nn_heads, head_scores_dict 