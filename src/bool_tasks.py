import math

import torch
import numpy as np
import pdb
from time import time


def squared_error(ys_pred, ys):
	return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
	return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
	return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
	'''
	ys_pred: [-inf, inf]
	ys: {-1, 1}
	'''
	output = sigmoid(ys_pred)
	target = (ys + 1) / 2
	return bce_loss(output, target)


class Task:
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
		self.n_dims = n_dims
		self.b_size = batch_size
		self.pool_dict = pool_dict
		self.seeds = seeds
		assert pool_dict is None or seeds is None

	def evaluate(self, xs):
		raise NotImplementedError

	@staticmethod
	def generate_pool_dict(n_dims, num_tasks):
		raise NotImplementedError

	@staticmethod
	def get_metric():
		raise NotImplementedError

	@staticmethod
	def get_training_metric():
		raise NotImplementedError


def get_task_sampler(
	task_name, n_dims, batch_size, n_points, pool_dict=None, num_tasks=0, **kwargs
):
	task_names_to_classes = {
		"conjunction": Conjunction,
		'teach_biconjunction': TeachBiConjunction,
		"mono_conjunction": MonoConjunction,
		"teach_conjunction": TeachConjunction,
		"disjunction": Disjunction,
		"sparse_disjunction": SparseDisjunction,
		"nearest_neighbours": NearestNeighbours,
		"parity": Parity,
		"sparse_parity": SparseParity,
		"majority": Majority,
		"int_halfspace": IntHalfspace,
		"dnf": DNF,
		"teach_dnf": TeachDNF,
		"cnf": CNF,
		'sparse_thres': SparseThreshold,
	}
	if task_name in task_names_to_classes:
		task_cls = task_names_to_classes[task_name]
		if num_tasks > 0:
			if pool_dict is not None:
				raise ValueError("Either pool_dict or num_tasks should be None.")
			print('Generating pool dict for {} tasks'.format(num_tasks))
			if task_name in ['conjunction', 'majority', 'disjunction', 'sparse_parity', 'dictator', 'sparse_disjunction', 'nearest_neighbours']:
				print('Generating pool dict for {}'.format(task_name))
				pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, n_points, **kwargs)
			else:
				pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
		return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
	else:
		print("Unknown task")
		raise NotImplementedError






class Conjunction(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(Conjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		k = int(n_dims/3)
		if pool_dict is None:
			self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
			self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
			self.xs_b = None
		else:
			assert 'w' in pool_dict
			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
			self.w_b = pool_dict["w"][indices]
			self.kw = pool_dict["kw"][indices]
			self.xs_b = pool_dict["xs"][indices]
	
	def sample_xs(self, n_points, b_size):
		if self.xs_b is not None:
			# Using pre-generated xs
			return self.xs_b
		else:
			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

			for b in range(b_size):
				wb, k = self.w_b[b], self.kw[b]            
				pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
				nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
				for i in range(n_points):
					if np.random.choice([0, 1], p=[0.7, 0.3]):
						xs_b[b, i, pidx] = +1.0
						xs_b[b, i, nidx] = -1.0
						assert (xs_b[b, i, :] @ wb).squeeze() >= k

			return xs_b


	@staticmethod
	def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
		# w_b shape: (num_tasks, n_dims, 1)
		start_time = time()

		w_b = torch.tensor(np.random.choice([0, 1, -1], size=(num_tasks, n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		kw = torch.norm(w_b, p=1, dim=1) - 1

		xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1

		for b in range(num_tasks):
			wb, k = w_b[b], kw[b]            
			pidx = [i for i in range(n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(n_dims) if wb[i] == -1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.7, 0.3]):
					xs_b[b, i, pidx] = +1.0
					xs_b[b, i, nidx] = -1.0
					assert (xs_b[b, i, :] @ wb).squeeze() >= k

		end_time = time()
		print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))

		return {"w": w_b, "kw": kw, "xs": xs_b}
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy





class TeachBiConjunction(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(TeachBiConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		for b in range(b_size):
			wb, k = self.w_b[b], self.kw[b]            
			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.6, 0.4]):
					xs_b[b, i, pidx] = +1.0
					xs_b[b, i, nidx] = -1.0
					assert (xs_b[b, i, :] @ wb).squeeze() >= k
		
		# Adding teaching sequence in the beginning of samples

		for b in range(b_size):
			wb = self.w_b[b].squeeze()
			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
			ex  = len(pidx) + len(nidx) + 2
			new_ex  = wb.repeat(ex, 1)  # new_ex shape: (ex, n_dims)

			for i in range(self.n_dims):
				if i not in pidx and i not in nidx:
					new_ex[0, i] = -1.0

			for i in range(self.n_dims):
				if i not in pidx and i not in nidx:
					new_ex[1, i] = 1.0

			for k in range(2, ex):
				for i in range(self.n_dims):
					if i not in pidx and i not in nidx:
						new_ex[k, i] = -1.0


			cx = 2
			for i in range(len(pidx)):
				new_ex[cx, pidx[i]] = -1.0
				cx += 1
			
			for i in range(len(nidx)):
				new_ex[cx, nidx[i]] = 1.0
				cx += 1
			
			assert cx == ex
			

			# idx = torch.randperm(len(new_ex))
			# new_ex = new_ex[idx]
			xs_b[b, 0:ex, :] = new_ex


			

		return xs_b

		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy




	



class MonoConjunction(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
		super(MonoConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

		self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[2/3, 1/3]), dtype=torch.float)
		self.kw = self.w_b.sum(dim=1) - 1
	
	def sample_xs(self, n_points, b_size):

		xs_b = torch.tensor(np.random.choice([0, 1], size=(b_size, n_points, self.n_dims), p=[1-self.p, self.p]), dtype=torch.float)*2-1 

		return xs_b

		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	



class TeachConjunction(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(TeachConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)
		self.kw = self.w_b.sum(dim=1) - 1
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		for b in range(b_size):
			wb, k = self.w_b[b], self.kw[b]
			tidx = [i for i in range(self.n_dims) if wb[i] == 1]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.6, 0.4]):
					xs_b[b, i, tidx] = +1.
					assert (xs_b[b, i, :] @ wb).squeeze() >= k
		
		# Adding teaching sequence in the beginning of samples
		for b in range(b_size):
			wb = self.w_b[b].squeeze()
			tidx = [i for i in range(self.n_dims) if wb[i] == 1]
			ex  = len(tidx) + 1
			new_ex  = wb.repeat(ex, 1)
			for i in range(len(tidx)):
				cx = i+1
				new_ex[cx, tidx[i]] = 0
			
			new_ex = new_ex * 2 - 1

			# idx = torch.randperm(len(new_ex))
			# new_ex = new_ex[idx]
			xs_b[b, 0:ex, :] = new_ex
			

		return xs_b

		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	




class Disjunction(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(Disjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		if pool_dict is None:
			self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
			self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
			self.xs_b = None
		else:
			assert 'w' in pool_dict
			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
			self.w_b = pool_dict["w"][indices]
			self.kw = pool_dict["kw"][indices]
			self.xs_b = pool_dict["xs"][indices]
			
	
	def sample_xs(self, n_points, b_size):
		if self.xs_b is not None:
			# Using pre-generated xs
			return self.xs_b
		else:
			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1


			# Manipulate the input to create negative examples to make a more balanced dataset
			for b in range(b_size):
				wb, k = self.w_b[b], self.kw[b]            
				pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
				nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
				for i in range(n_points):
					if np.random.choice([0, 1], p=[0.7, 0.3]):
						xs_b[b, i, pidx] = -1.0
						xs_b[b, i, nidx] = +1.0
						assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

			return xs_b


	@staticmethod
	def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
		# w_b shape: (num_tasks, n_dims, 1)
		start_time = time()
		w_b = torch.tensor(np.random.choice([0, 1, -1], size=(num_tasks, n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		kw = torch.norm(w_b, p=1, dim=1) - 1

		xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1

		for b in range(num_tasks):
			wb, k = w_b[b], kw[b]            
			pidx = [i for i in range(n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(n_dims) if wb[i] == -1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.7, 0.3]):
					xs_b[b, i, pidx] = -1.0
					xs_b[b, i, nidx] = +1.0
					assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

		end_time = time()
		print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))


		return {"w": w_b, "kw": kw, "xs": xs_b}
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() + self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy






class SparseDisjunction(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(SparseDisjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)

		self.k = 4
		if pool_dict is None:
			wb = []
			for i in range(self.b_size):
				idx = np.random.choice(range(self.n_dims), self.k, replace=False)
				w = np.zeros(self.n_dims)
				w[idx] = 1
				wb.append(w)
			
			wb = np.array(wb)
			self.w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
			self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
			self.xs_b = None
			
			
			
		else:
			assert 'w' in pool_dict
			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
			self.w_b = pool_dict["w"][indices]
			self.kw = pool_dict["kw"][indices]
			self.xs_b = pool_dict["xs"][indices]
			
	
	def sample_xs(self, n_points, b_size):
		if self.xs_b is not None:
			# Using pre-generated xs
			return self.xs_b
		else:
			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1


			# Manipulate the input to create negative examples to make a more balanced dataset
			for b in range(b_size):
				wb, k = self.w_b[b], self.kw[b]            
				pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
				nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
				for i in range(n_points):
					if np.random.choice([0, 1], p=[0.7, 0.3]):
						xs_b[b, i, pidx] = -1.0
						xs_b[b, i, nidx] = +1.0
						assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

			return xs_b


	@staticmethod
	def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
		# w_b shape: (num_tasks, n_dims, 1)
		start_time = time()
		k = 4
		wb = []
		for i in range(num_tasks):
			idx = np.random.choice(range(n_dims), k, replace=False)
			w = np.zeros(n_dims)
			w[idx] = 1
			wb.append(w)
		
		wb = np.array(wb)
		w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
		kw = torch.norm(w_b, p=1, dim=1) - 1


		xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1

		for b in range(num_tasks):
			wb, k = w_b[b], kw[b]            
			pidx = [i for i in range(n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(n_dims) if wb[i] == -1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.7, 0.3]):
					xs_b[b, i, pidx] = -1.0
					xs_b[b, i, nidx] = +1.0
					assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

		end_time = time()
		print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))


		return {"w": w_b, "kw": kw, "xs": xs_b}
		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() + self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy



class NearestNeighbours(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, start_idx=0):
		super(NearestNeighbours, self).__init__(n_dims, batch_size, pool_dict, seeds)
		self.start_idx = start_idx

		if pool_dict is None:
			self.xs_b = None
		else:
			indices = torch.randperm(len(pool_dict["xs"]))[:batch_size]
			self.xs_b = pool_dict["xs"][indices]
			# self.ys_b = pool_dict["ys"][indices]
	# def check_unique(self, xs_b):
	# 	temp_xs = xs_b[:, :self.start_idx, :] # bs x start_idx x n_dims
	# 	temp_xs_2d = temp_xs.reshape(-1, temp_xs.shape[2]) # bs * n_points x n_dims
	# 	_, inverse_indices = torch.unique(temp_xs_2d, dim=0, return_inverse=True)
	# 	inverse_indices = inverse_indices.reshape(temp_xs.shape[0], temp_xs.shape[1]) # bs x start_idx
	# 	for row in inverse_indices:
	# 		if len(torch.unique(row)) != self.start_idx:
	# 			return False
			
	# 	return True
	
	def sample_xs(self, n_points, b_size):

		# xs_b = None
		# unique_found = False
		# while(not unique_found):
		if self.xs_b is not None:
			return self.xs_b
		else:
			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
			# unique_found = self.check_unique(xs_b)
		return xs_b
		
	def evaluate(self, xs_b):
		# if self.ys_b is not None:
		# 	return self.ys_b
		# else:
		ys_b = torch.randint(0, 2, (xs_b.shape[0], self.start_idx), dtype= torch.float)*2-1 # bs x start_idx

		xs_norm = torch.norm(xs_b, dim=2, keepdim=True)
		xs_normalized = xs_b / xs_norm
		xs_T = torch.transpose(xs_normalized, 1, 2) # bs x n_dims x n_points
		sim_mx = torch.matmul(xs_normalized, xs_T) # bs x n_points x n_points

		for pt in range(1, self.start_idx): # across initial points
			for batch in range(xs_b.shape[0]): # across batch
				similarities = sim_mx[batch][pt][:pt] # consider similarities with tensors occuring before
				similarities = torch.round(similarities, decimals=7)
				selected_idx = torch.argmax(similarities)
				if similarities[selected_idx].item() > 0.9999:
					# if ys_b[batch][selected_idx] != ys_b[batch][pt]:
					# 	pdb.set_trace()
					ys_b[batch][pt] = ys_b[batch][selected_idx]

		for pt in range(self.start_idx, xs_b.shape[1]): # across points
			y_vals = []
			for batch in range(xs_b.shape[0]): # across batch
				similarities = sim_mx[batch][pt][:pt] # consider similarities with tensors occuring before
				try:
					similarities = torch.round(similarities, decimals=7)
					selected_idx = torch.argmax(similarities)
				except:
					pdb.set_trace()
				y_vals.append(ys_b[batch][selected_idx].item())
			y_col = torch.tensor(y_vals).unsqueeze(1)
			ys_b = torch.cat((ys_b, y_col), dim=1)

		return ys_b

	@staticmethod
	def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
		# w_b shape: (num_tasks, n_dims, 1)
		start_time = time()

		xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1
		# ys_b = self.evaluate(xs_b)

		end_time = time()
		print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))

		return {"xs": xs_b}

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy



class SparseThreshold(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(SparseThreshold, self).__init__(n_dims, batch_size, pool_dict, seeds)

		self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
		thres_bound = 3
		self.kw = torch.randint(-thres_bound, thres_bound, (self.b_size, 1),  dtype= torch.float) + 0.5
		
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		return xs_b

		
	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - self.kw
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy






class IntHalfspace(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(IntHalfspace, self).__init__(n_dims, batch_size, pool_dict, seeds)
		bound = 3
		self.w_b = torch.randint(-bound, bound+1, (self.b_size, self.n_dims, 1),  dtype= torch.float)

	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
		return xs_b

	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - 0.5
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy





class Majority(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(Majority, self).__init__(n_dims, batch_size, pool_dict, seeds)

		if pool_dict is None:
			self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)
			self.xs_b = None
		else:
			assert 'w' in pool_dict
			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
			self.w_b = pool_dict["w"][indices]
			self.xs_b = pool_dict["xs"][indices]

	
	def sample_xs(self, n_points, b_size):
		if self.xs_b is not None:
			# Using pre-generated xs
			return self.xs_b
		else:
			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

			return xs_b


	@staticmethod
	def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
		# w_b shape: (num_tasks, n_dims, 1)

		w_b = torch.tensor(np.random.choice([0, 1], size=(num_tasks, n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)

		xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1


		return {"w": w_b, "xs": xs_b}

	def evaluate(self, xs_b):
		w_b = self.w_b.to(xs_b.device)
		ys_b = (xs_b @ w_b).squeeze() - 0.5
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy
	







class Parity(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(Parity, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		# Approximate 35% of indices will be 1
		funcs = np.random.choice(2**n_dims, size = batch_size)
		all_subsets  = self.generate_subsets(n_dims)
		self.w_b = torch.zeros(size= (batch_size, n_dims, 1))
		# self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
		for i in range(batch_size):
			self.w_b[i, all_subsets[funcs[i]]] = 1
		
	
	def sample_xs(self, n_points, b_size):
		# Input distribution is uniform over {-1, 1}^n_dims
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		return xs_b

		
	def evaluate(self, xs_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = self.w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		return ys_b.sign()


	def generate_subsets(self, n):
		subsets = []
		for i in range(2**n):
			subset = [j for j in range(n) if (i & 1 << j)]
			subsets.append(subset)
		return subsets
	
	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy







class SparseParity(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(SparseParity, self).__init__(n_dims, batch_size, pool_dict, seeds)

		if pool_dict is None:
			self.k = 2
			wb = []
			for i in range(self.b_size):
				idx = np.random.choice(range(self.n_dims), self.k, replace=False)
				w = np.zeros(self.n_dims)
				w[idx] = 1
				wb.append(w)
			
			wb = np.array(wb)
			self.w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
			self.xs_b = None
		else:
			assert 'w' in pool_dict
			indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
			data_indices = torch.randperm(len(pool_dict["xs"]))[:batch_size]
			self.w_b = pool_dict["w"][indices]
			self.xs_b = pool_dict["xs"][data_indices]
		# self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
		
	
	def sample_xs(self, n_points, b_size):
		if self.xs_b is not None:
			# Using pre-generated xs
			return self.xs_b
		else:
			# Input distribution is uniform over {0, 1}^n_dims
			xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

			return xs_b

		
	def evaluate(self, xs_b):
		# Output \in {-1, 1}
		xt = (xs_b.clone() +1)/2
		w_b = self.w_b.to(xs_b.device)
		ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
		return ys_b.sign()

	@staticmethod
	def generate_pool_dict(n_dims, num_tasks, n_points, **kwargs):
		start_time = time()
		k=2
		wb = []
		num_funcs = (math.comb(n_dims, k) //2) +1  # Different than other function classes
		print('Sampling {} parity functions'.format(num_funcs))
		for i in range(num_funcs):
			idx = np.random.choice(range(n_dims), k, replace=False)
			w = np.zeros(n_dims)
			w[idx] = 1
			wb.append(w)
		
		wb = np.array(wb)
		w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)

		xs_b = torch.randint(0, 2, (num_tasks, n_points, n_dims), dtype= torch.float)*2-1

		end_time = time()
		print('Time to generate pool dict: {:.2f} mins {:.2f} secs'.format((end_time-start_time)//60, (end_time-start_time)%60))

		return {"w": w_b, "xs": xs_b}

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy











# Three DNF Task named DNF for simplicity. Complete DNF is hard to learn complexity-wise, so we use a 3-term DNF
class DNF(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(DNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		self.w_b = [torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.8, 0.1, 0.1]), dtype=torch.float) for i in range(3)] # Create 3 clauses
		self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		# Manipulate the input to create positive examples to make a more balanced dataset
		for b in range(b_size):
			cid = np.random.choice([0, 1, 2])        # Choose a clause
			wb, k = self.w_b[cid][b], self.kw[cid][b]
			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.65, 0.35]):
					xs_b[b, i, pidx] = +1.0
					xs_b[b, i, nidx] = -1.0
					assert (xs_b[b, i, :] @ wb).squeeze() >= k

		return xs_b

		
	def evaluate(self, xs_b):
		w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
		ys_bs = [(xs_b @ w_bs[i]).squeeze() - self.kw[i] for i in range(3)]
		ys_bs = [ys_bs[i].sign() for i in range(3)]
		# Combine stack three tensors into one
		ys_b = torch.stack(ys_bs, dim=2).max(dim=2)[0]  # 0th Index is the value, 1st index has indices
		
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy





# Three DNF Task named DNF for simplicity. Complete DNF is hard to learn complexity-wise, so we use a 3-term DNF
class TeachDNF(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(TeachDNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		self.w_b = [torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.8, 0.2]), dtype=torch.float) for i in range(3)] # Create 3 clauses
		self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		# Manipulate the input to create positive examples to make a more balanced dataset
		for b in range(b_size):
			cid = np.random.choice([0, 1, 2])        # Choose a clause
			wb, k = self.w_b[cid][b], self.kw[cid][b]
			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.65, 0.35]):
					xs_b[b, i, pidx] = +1.0
					assert (xs_b[b, i, :] @ wb).squeeze() >= k
		
		# Adding teaching sequence in the beginning of samples
		for b in range(b_size):
			wb_f = [self.w_b[i][b].squeeze() for i in range(3)]
			tidxs = []
			for wb in wb_f:
				pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
				tidxs.append(pidx)
			
			prev_ex_len = 0
			for k in range(len(wb_f)):
				
				wb = wb_f[k]
				tidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
				ex = len(tidx) + 1
				new_ex = wb.repeat(ex, 1)
				
				for i in range(len(tidx)):
					cx = i+1
					try:
						new_ex[cx, tidx[i]] = 0
					except:
						pdb.set_trace()
				
				new_ex = new_ex * 2 -1

				xs_b[b, prev_ex_len: prev_ex_len + ex, :] = new_ex
				prev_ex_len += ex
				# xs_b[b, ex_lens[k-1]:ex_lens[k-1]+ex, :] = new_ex

		

		return xs_b

		
	def evaluate(self, xs_b):
		w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
		ys_bs = [(xs_b @ w_bs[i]).squeeze() - self.kw[i] for i in range(3)]
		ys_bs = [ys_bs[i].sign() for i in range(3)]
		# Combine stack three tensors into one
		ys_b = torch.stack(ys_bs, dim=2).max(dim=2)[0]  # 0th Index is the value, 1st index has indices
		
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy








class CNF(Task):
	def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
		super(CNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
		# self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
		self.w_b = [torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.80, 0.1, 0.1]), dtype=torch.float) for i in range(3)] # Create 3 clauses
		self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
	
	def sample_xs(self, n_points, b_size):
		xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

		
		# Manipulate the input to create negative examples to make a more balanced dataset
		for b in range(b_size):
			cid = np.random.choice([0, 1, 2])        # Choose a clause
			wb, k = self.w_b[cid][b], self.kw[cid][b]
			pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
			nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
			for i in range(n_points):
				if np.random.choice([0, 1], p=[0.7, 0.3]):
					xs_b[b, i, pidx] = -1.0
					xs_b[b, i, nidx] = +1.0
					assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

		return xs_b

		
	def evaluate(self, xs_b):
		w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
		ys_bs = [(xs_b @ w_bs[i]).squeeze() + self.kw[i] for i in range(3)]
		ys_bs = [ys_bs[i].sign() for i in range(3)]
		# Combine stack three tensors into one
		ys_b = torch.stack(ys_bs, dim=2).min(dim=2)[0]  # 0th Index is the value, 1st index has indices
		
		return ys_b.sign()

	@staticmethod
	def get_metric():
		return accuracy

	@staticmethod
	def get_training_metric():
		return cross_entropy

